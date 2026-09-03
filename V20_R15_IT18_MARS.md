# V20 R15 IT18 — MARS (MORIARTY, standing adversary)

Target: the three instruments this round shipped in it.16–it.17, each certified by
its own author. One runnable VALUE test per strike, RED first against unmutated
code, under `tests/mars_v20/`.

Search proof (an invocation finding a string known to exist):

```
$ grep -n "cannot corroborate either way" scripts/iteration_timer.sh
157:      ANY_DEAD=1   # nothing registered: cannot corroborate either way, say so below
grep exit=0
```

---

## STRIKE 1 — the heartbeat repair's corroboration is vacuous, and the `--force` licence was moved, not removed

**Instrument:** `C:\Users\seal\Desktop\New folder (32)\scripts\iteration_timer.sh`, `check` branch.
**Test:** `C:\Users\seal\Desktop\New folder (32)\tests\mars_v20\test_it18_timer_liveness_defaults_permissive.py`
**Status: 3 failed in 1.18s — RED against unmutated code, no mutation applied.**

```
$ python -m pytest tests/mars_v20/test_it18_timer_liveness_defaults_permissive.py -q
FAILED ...::test_missing_pidfile_does_not_assert_a_process_fact
FAILED ...::test_absent_and_empty_pidfile_agree
FAILED ...::test_force_rearm_licence_is_actually_gone
3 failed in 1.18s   (exit 1)
```

### 1a. The corroboration never runs, and defaults to the struck answer

`.claude/iteration.pids` **does not exist**:

```
$ ls -la .claude/iteration.pids
ls: cannot access '.claude/iteration.pids': No such file or directory
```

Nothing in the repo writes it. The script's own header says the watchdog "kills
only PIDs written to `.claude/iteration.pids` **by the caller**" — and there is no
caller. So the corroboration branch the repair added is dead code, and control
falls to:

```sh
else
  ANY_DEAD=1   # nothing registered: cannot corroborate either way, say so below
fi
if [[ $GAP -gt 300 && $ANY_DEAD -eq 1 ]]; then
```

`ANY_DEAD=1` is the *permissive* value. The repair's answer to "no evidence" is
the same answer it gives to "evidence of death". The Inspector's strike was that
every overrun past `cap + 5m` reads as an interruption; after the repair, in the
only state this repo is ever in, **every check gap past 5m still reads as an
interruption**. Verbatim, from the sandboxed run with no pidfile:

```
iteration 18: 10m0s WALL, but 10m0s of it had NO live process.
   Wall clock is not work -- but this is a HINT, not a verdict: no registered pid
   was alive to corroborate. Confirm the gap was an interruption BEFORE re-arming.
```

The headline sentence is a flat assertion about processes the script never
inspected. The hedge is on line 2; the number is on line 1.

### 1b. Absent and empty pidfile give opposite verdicts

Same evidentiary state — nothing registered — two spellings, two answers, because
the `else` fires on a missing **file**, not a missing **pid**:

```
--- absent (no iteration.pids) ---
iteration 18: 10m0s WALL, but 10m0s of it had NO live process.
--- empty (iteration.pids exists, zero bytes) ---
iteration 18: 10m0s elapsed, 10m0s left of 20m
```

`touch .claude/iteration.pids` silences the interruption verdict permanently.
`rm` it and every gap is an interruption. Neither spelling registered a pid.

### 1c. The `--force` offer was deleted from the wrong file

The repair removed `--force` from the OVERDUE branch's printed text. `start`
still accepts it (`[[ "${2:-}" != "--force" ]]`), and the **standing
per-iteration instruction**, re-read every turn, still says — verbatim from
`C:\Users\seal\Desktop\New folder (32)\.claude\ralph-loop.local.md`:

> If check reports dead time with no live process, that is an INTERRUPTION not an
> overrun - re-arm with --force and continue.

That sentence is keyed to the exact string 1a emits unconditionally. The chain is
closed: no pidfile → `ANY_DEAD=1` → "no live process" → standing order to
`start --force` → fresh 20 minutes. The Inspector struck the licence in the
script; it was never in the script's gift to remove.

### Replacement route — RETIRE the pid channel, REPRICE the beat

1. Delete the `PIDFILE` corroboration entirely. It has no writer; a check nobody
   feeds is worse than no check, because it launders "unknown" as "confirmed".
2. Make the beat honest about what it measures. Rename the output to what it is:
   `iteration 18: 10m0s elapsed; 10m0s since the last check call`. No claim about
   processes, no "WALL", no "NO live process".
3. Delete the `--force` sentence from `.claude/ralph-loop.local.md` in the same
   commit as any change to the timer's advisory text, or the strike survives the
   repair a third time.
4. If liveness is genuinely wanted, the writer must exist before the reader:
   have the dispatching planet `echo $$ >> .claude/iteration.pids`, and treat a
   missing/empty pidfile as `ANY_DEAD=0` (unknown ⇒ no licence), not `1`.

---

## STRIKE 1, SUPPLEMENT — one finished nurse licenses the interruption verdict

Black-box matrix over the sandboxed timer, `start=NOW-600 budget_s=1200
beat=NOW-600` (10m elapsed of 20m, 10m check gap), one variable: the pidfile.
All exit 0 unless noted. Verbatim:

| case | `.claude/iteration.pids` | verdict |
|---|---|---|
| A | absent | `10m5s WALL, but 10m0s of it had NO live process.` |
| B | exists, empty | `10m6s elapsed, 9m54s left of 20m` |
| C | one LIVE pid | `10m5s elapsed, 9m55s left of 20m` |
| D | one DEAD pid | `... had NO live process.` |
| **E** | **one live pid AND one dead pid** | **`... had NO live process.`** |
| F | absent, start=NOW-1500 | `OVERDUE ... 25m4s against a 20m cap` (exit 1) |
| G | `printf '1234'`, no trailing newline | `10m5s elapsed, 9m55s left of 20m` |

**Case E is the operating condition.** `ANY_DEAD` is set by *any* dead pid and
never cleared, so once a single nurse of four exits, every subsequent check gap
over 5m announces "NO live process" while three nurses are running. The predicate
the repair chose is `∃ dead`, and the question it answers is `∀ dead`.

**Case G** is the same bug in the reader: `while read -r p` drops a final line
with no newline, so a pidfile written with `printf`/`echo -n` registers nothing
and silently reads as case B. The three ways to have no usable pid (absent,
empty, unterminated) split 1–2 across the two opposite verdicts.

Add to the replacement route: `ANY_DEAD` must be `all pids dead AND at least one
pid registered`, and the read must be `while read -r p || [[ -n $p ]]`.

---

## STRIKE 2 — the clause-(b) repair certifies the manifest against itself

**Instrument:** `C:\Users\seal\Desktop\New folder (32)\tests\saturn\test_v20_r15_freeze_manifest.py`, `_arm_of()` (lines 304–311) and `test_every_clause_b_line_journals_that_wings_own_arm()` (314–331).
**Test:** `C:\Users\seal\Desktop\New folder (32)\tests\mars_v20\test_it18_wing_identity_is_self_certified.py`
**Status: 1 failed, 1 passed in 0.65s — `assert []`, RED against unmutated code.**

SATURN's repair derives the arm name from the wing's own clause-(a) module path,
reasoning that "a hardcoded map would be a second place for the manifest to be
wrong". A hardcoded map is also the only place the manifest can be *caught* being
wrong. After the repair clause (b) is checked against clause (a), and clause (a)
is checked against nothing. The wing→arm binding is a closed loop inside one
file: the manifest can no longer contradict itself about a wing's identity, only
be uniformly wrong about it.

### The falsifying mutation, and its V-26 receipt

Exchange W1's and W3's clause-(a) **and** clause-(b) citations. Nothing else
moves. `test_the_mutation_is_statistic_preserving` (PASSES) asserts the
precondition: same row count, same wing multiset, same `(wing, clause)` key set,
same multiset of citations. Every per-collection statistic is preserved.

After the swap `_arm_of("W1") == "arm_pl"` and `_arm_of("W3") == "arm_smprime"`.
Seven semantic nodes were then called on the mutated rows. **Zero fired:**

```
arm_of W1 = arm_pl   arm_of W3 = arm_smprime
PASS  clause_b ()                      <-- the repair itself
PASS  found_not_named ('arm_smprime',)
PASS  found_not_named ('arm_pl',)
PASS  resolves ('a',) ('b',) ('c',) ('d',)
PASS  only_found ()
PASS  four_clauses ()
PASS  no_dup_line ()
PASS  N ()
```

W1 — whose clauses (c) and (d) still cite `V16_ARM_SMPRIME.md:529` and the
`15.970` smprime price — is now certified as `arm_pl`, and the node SATURN built
to catch exactly that says GREEN. **The `kind` exchange he struck is still
reachable; you exchange clause (a) alongside it.**

Two contributing surfaces found while landing this:

* `test_every_frozen_wing_is_found_and_not_merely_named` is parametrized
  `["arm_smprime", "arm_pl"]` — on arm names, with no wing argument at all. It
  can never witness a wing↔arm mismatch; it only asserts both arms exist
  *somewhere* in `results/`.
* `FREEZE-SHA256` does not close this. The digest proves the list was not edited
  *after* freezing. It says nothing about the list being right *when* frozen, and
  a deliberate re-freeze recomputes it. Tamper-evidence was never identity
  evidence.

### Replacement route — REROUTE the binding to clauses (c)/(d)

The manifest already carries two citations per wing that point outside the arm
module: (c) a ledger/doc line, (d) a price. Bind them. `_arm_of` should require
**agreement across clause (a) and clause (c)** — derive the arm from (a), then
assert the (c) citation's path or anchor names that same arm — so the identity
rests on two independently authored files rather than one row. Cost: one
`assert`, and the W1 (c) row already satisfies it (`V16_ARM_SMPRIME.md` contains
`SMPRIME`). Under the swap, that assert fires.

---

## STANDING QUESTION — `ρ(beta) = −0.0324`: nothing has changed it

**Status: the sign is still one cell, and seed 2 is still the cell.** Swept
`house-events.jsonl`, the journal, and every it.16–it.18 report.

* The M12-B2 record stands unamended:
  `house-events.jsonl:12771` — *"rho(beta)=-0.0324 is sign-unstable to ONE cell.
  Dropping seed 2 (the single crossing cell) flips it to +0.089286 while rho(qk)
  survives at +0.657143 p=0.007770."*
* `V20_R15_JOURNAL.md:2433` and `:2493` restate it; `:1424` carries the
  recomputation; `V20_R15_IT567_INSPECTOR.md:315` carries the original.
* **No restatement, revision or withdrawal in `V20_R15_IT16_INSPECTOR.md`,
  `V20_R15_IT17_JUPITER.md`, `V20_R15_IT17_MERCURY.md` or
  `V20_R15_IT18_JUPITER.md`.**

`arm_smprime` being draw-checked 48/48 with zero flips does **not** touch this.
That result is about the *arm's* stability across draws; the −0.0324 instability
is about the **correlation's** dependence on one row of a 16-row table. Seed 2 is
the row: `results/v17k_r4_retake.jsonl` gives it `beta=+1.34393`,
`eval_nrmse=0.203920` against a field whose next-best nrmse is `0.852`. That seed
2 crosses on all three draws makes it *more* load-bearing, not less — it remains
simultaneously the whole of W1's remaining probability mass and the single row
whose deletion flips the sign of the statistic used to strike the M1 transfer.
Both facts resting on one cell is the exposure, and it is unchanged at it.18.

Test already standing and unmutated:
`tests/mars_v20/test_it12_the_four_constants.py::test_the_beta_rho_sign_is_decided_by_a_single_cell`.

---

## STRIKE 3 — "re-scoped" is the wrong verb. The claim was withdrawn.

**Instrument:** `C:\Users\seal\Desktop\New folder (32)\tests\jupiter\test_v20_r15_it14_theory_table.py::bad_citations` (lines 53–70), and its it.17 replacement `tests\jupiter\test_v20_r15_it17_citation_landing.py`.
**Test:** `C:\Users\seal\Desktop\New folder (32)\tests\mars_v20\test_it18_census_zero_bad_was_never_a_citation_claim.py`
**Status: 2 failed, 1 passed in 0.58s — RED against unmutated code. The second
failure (`test_the_landing_checker_covers_the_census_it_answers`) is superseded by
JUPITER's own it.18 filing and is withdrawn below; the strike rests on the first.**

Re-scoping is honest when the original number measured a real, narrower property.
The test is whether the narrower property was ever measured. Read `bad_citations`
mechanically: it appends to `bad` on exactly two conditions —

```python
if not p.is_file():                      bad.append(f"... — no such file")
if not (1 <= int(lineno) <= len(lines)): bad.append(f"... — file has {len(lines)} lines")
```

— a file-exists check and `1 <= n <= len(lines)`. **It never opens the cited
line.** There is no scope at which that is a statement about citations. `0 bad`
is a line-count result carried under a citation headline.

### The receipt is JUPITER's own

`test_v20_r15_it17_citation_landing.py::WITHDRAWN` lists **25 pointers JUPITER
himself certifies are wrong and has repaired**. All 25 were inside the 123. Fed
back to the it.14 checker:

```
withdrawn pointers: 25   named bad by it.14 checker: 0
```

**Zero of twenty-five.** A checker that passes every citation its own author has
withdrawn did not measure a narrower version of citation fitness. It measured a
different quantity. The five defect species JUPITER catalogued at it.17 —
rigid shift, adjacent row, `:1`-as-module, def-vs-body, wrong file entirely —
are *all* invisible to it, as his own §2.2 says: "None of the five is detectable
by asking whether the file exists."

### The half of this strike that JUPITER has already answered — corrected before filing

The first draft of this section read "the replacement certifies 30 of 123 and
inherits the headline", on the evidence that
`test_v20_r15_it17_citation_landing.py::MANIFEST` has 30 entries. **That is
superseded and the draft assertion is withdrawn.**
`tests\jupiter	est_v20_r15_it18_citation_landing.py` exists in the tree and
carries `POPULATION_AT_IT18 = 129`. Its docstring states that the it.17 manifest
"is the wrong population twice over" — the table now holds 128 citations because
two it.17 repairs replaced one citation with four and one with two, and the
manifest covered the citations it.17 *repaired*, not the citations it.17 *wrote*.
It ships three further failures, **two of them introduced by his own it.17
repairs**.

That is not re-scoping. That is a full re-census of the corrected population with
self-inflicted defects filed against himself, and it is the strongest form of the
answer. **The coverage half of this strike does not fire.**

### Verdict on the verb

**WITHDRAW, not re-scope.** The distinction matters because a re-scoped claim
keeps its instrument and its GREEN; a withdrawn claim surrenders both. Concretely:

1. Rename `bad_citations` to `unresolvable_citations` and its node to
   `test_no_citation_points_outside_its_file`. The function is fine; the name and
   the headline are the defect.
2. Delete `123/123, 0 bad` from every headline that survives, or write it as two
   numbers, because there are two measurements: `N resolve; M verified to land`.
3. Carry `POPULATION_AT_IT18 = 129` forward as the population of record. `123` is
   now wrong on both axes — it is neither the current count nor the set the
   Inspector's own finding describes (the census spans 37 files while the table
   carries 104 instances / 82 unique).

**Credit where it is owed:** the it.17 instrument's refusal to generate its
manifest from the table it checks ("a manifest derived from the thing it checks
asserts nothing") is the correct principle, and it is the principle STRIKE 2
found missing from the wing manifest. The defect here is a headline that outlived
its measurement, not the work under it.

---

## ATTACKS THAT DID NOT FIRE

Reported because a strike list without its misses is a filtered one.

1. **The freeze hash as an identity check (STRIKE 2, first approach).** The
   working hypothesis was that swapping W1/W3 clause (a)+(b) would be caught only
   by `FREEZE-SHA256`, making the semantic layer decorative. It is caught by the
   hash — but a deliberate re-freeze recomputes the hash, so the hash was never
   the target. Dropped as an argument; kept as a one-line note under STRIKE 2.

2. **`test_every_frozen_wing_is_found_and_not_merely_named` as the catcher.** My
   first harness passed it `("W1",)` and `("W3",)` and it fired — and I nearly
   filed "the suite catches the swap" as a *failed* strike on that basis. It is
   parametrized on **arm names** (`["arm_smprime", "arm_pl"]`), not wing IDs, so
   `journalled_cells("W1") == 0` fired on my own bad call, not on the mutation.
   Corrected before filing; the corrected run is `assert []`. **The node is worse
   than I first thought — it takes no wing argument at all and therefore cannot
   witness a wing↔arm mismatch in either direction.**

3. **`kill -0` under Git Bash against a native Windows pid.** Suspected the
   liveness probe could not see non-msys processes at all, which would make even
   a populated pidfile meaningless. Cases C/D/E show `kill -0` discriminating live
   from dead correctly on bash-spawned pids. Not pursued further; unresolved for
   pids spawned outside bash, and moot while nothing writes the file.

4. **The OVERDUE branch's gap note.** Checked whether the OVERDUE path still
   emits an uncorroborated interruption hint. It does (case F: `NOTE: 10m4s since
   the last check. That MAY be an interruption.` with no pid consulted), but the
   repair's added sentences — "It is not a licence to re-arm" and the removal of
   the `--force` command — make that branch materially better than the strike
   found it. **The repair holds on the OVERDUE branch.** It fails on the
   non-OVERDUE branch and in `.claude/ralph-loop.local.md`, which is STRIKE 1.

5. **The census coverage gap.** Drafted as "30 of 123 certified, 93 unmeasured"
   and withdrawn on reading `tests/jupiter/test_v20_r15_it18_citation_landing.py`,
   which re-censuses the corrected population (`POPULATION_AT_IT18 = 129`) and
   files three further failures, two of them introduced by JUPITER's own it.17
   repairs. **He was ahead of the attack.** The verb finding stands; the coverage
   finding does not.

6. **`ρ(beta)` moved by the it.17 draw work.** Swept `house-events.jsonl`, the
   journal, and all four it.16–it.18 office reports for any restatement. None
   exists. The 48/48 draw-check does not bear on it. The attack found nothing to
   strike because the number has not moved — which is itself the answer to the
   standing question.

---

## SCOREBOARD

| # | target | test | status |
|---|---|---|---|
| 1 | timer heartbeat repair | `tests/mars_v20/test_it18_timer_liveness_defaults_permissive.py` | **3 failed** — RED |
| 2 | wing manifest clause-(b) repair | `tests/mars_v20/test_it18_wing_identity_is_self_certified.py` | **1 failed, 1 passed** — RED (`assert []`) |
| 3 | census standard | `tests/mars_v20/test_it18_census_zero_bad_was_never_a_citation_claim.py` | **2 failed, 1 passed** — RED; strike rests on failure 1 (`0 of 25 withdrawn pointers named`). Failure 2 withdrawn, superseded by `tests/jupiter/test_v20_r15_it18_citation_landing.py` |
| — | `ρ(beta) = −0.0324` | `tests/mars_v20/test_it12_the_four_constants.py` (standing) | unchanged at it.18 |

No mutation was left in the tree. No git write was performed. Nothing touched
Kaggle. All three strike tests are RED against unmutated code with no `and False`.
