# V13 — D-3 loop forensics

Read performed at `ecbedf7` (`feat/r9-causal-consequence`), working tree carrying
one modification (`results/r10_it8_waveB.log`). No file in this repository was
altered to produce it except this one.

---

## Cause of death

The loop did not terminate, it was removed: `max_iterations: 0` disables the stop
hook's only iteration ceiling
(`~/.claude/plugins/cache/claude-plugins-official/ralph-loop/1.0.0/hooks/stop-hook.sh:61`)
and the `completion_promise` demanded two 40-iteration ranges whose second cannot
begin until the first reaches it.35 (`PHASE2_CONTRACT_V_MAIN_4.md:22-30`), so no
exit condition could ever fire, and at ralph turn 600 — immediately after the
it.21 wave read `NO READING` at `n=8,192` and journalled its remaining step
budget as `"unaffordable, see priced DAG"`
(`results/r10_r10_it21_t8_n8192_capacity_softmax_t8.jsonl:8`) — the state file was
renamed by hand out of the hard-coded path `.claude/ralph-loop.local.md`, whose
existence is the only thing `stop-hook.sh:13-18` tests, and it is that rename, not
the `active: false` which `ecbedf7` credits and which the hook never parses
(`stop-hook.sh:21-25`), that stopped it.

Short form for a mount request: **the loop ran out of affordable work, not of
iterations, and was killed by hand because it had no ceiling of its own.**

---

## Deactivation forensics

### The `active` field is inert

`stop-hook.sh:21-25` parses exactly four frontmatter fields — `iteration`,
`max_iterations`, `completion_promise`, `session_id`. The string `active` occurs
in the plugin only in comments (`stop-hook.sh:4`, `:12`, `:16`) and in the writer
that emits it (`scripts/setup-ralph-loop.sh:179`). Nothing reads it.

The hook's complete set of exit paths:

| path | line | fires here? |
|---|---|---|
| state file absent | `:15-18` | **yes — this is the one that fired** |
| `session_id` mismatch | `:33-35` | no |
| `iteration` non-numeric | `:38-47` | no (`600`) |
| `max_iterations` non-numeric | `:49-58` | no (`0`) |
| `iteration >= max_iterations` | `:61-65` | **disabled** — guarded by `MAX_ITERATIONS -gt 0` |
| transcript missing / unparseable | `:70-126` | no |
| `<promise>` match | `:129-141` | no |

Setting `active: false` therefore changed nothing about the loop's behaviour. The
rename did all the work, and the plugin's own cancel path
(`commands/cancel-ralph.md:17`) is `rm .claude/ralph-loop.local.md` — the rename is
that `rm` performed non-destructively.

### The deactivation commit's diff does not contain the deactivation

`git show --stat ecbedf7` reports one changed file:
`results/r10_it8_waveB.log`, binary, 40068 → 40194 bytes, `1 file changed, 0
insertions(+), 0 deletions(-)`. The commit body asserts the loop file "now reads
`active: false`". No loop file appears in the diff.

The reason is structural, not a slip. `.claude/` is excluded at
`.git/info/exclude:7` (with `.claude/worktrees/` at `:8`), so
`git ls-files .claude/` returns empty and `git check-ignore -v` resolves both
`.claude/ralph-loop.local.md` and `.claude/ralph-loop.R10.stopped.md` to that
line. A full-history scan (`git ls-tree -r` across every ref) finds
`ralph-loop.R10.stopped.md` in **zero commits**. The R10 loop file was never
tracked and could not have been committed.

The last tracked appearance of any loop file is `9e1390c` (2026-08-26 11:39:52),
which **deleted** `.claude/ralph-loop.local.md` (`10 ----------`) as part of a
normal R6-era loop completion. Every prior add/delete pair — `dfc1591`/`9dcc25a`,
`2e1359c`/`d08f2fc`, `138251a`/`9e1390c` — is the plugin's own write-then-`rm`
cycle, committed. The exclusion was added to `.git/info/exclude` some time between
2026-08-26 11:39 and the R10 launch on 2026-08-30, and after that point the loop
state left no git trace at all.

**Recorded as a finding in its own right:** a commit message describing a change
its diff does not contain. This is MISTAKES.md `P-2` — *"A number whose only home
is a commit message"* (`MISTAKES.md:300-314`) — in its state-change form. The
deactivation is real and is verifiable on disk; it is not verifiable from the
history, and `ecbedf7` is the only record that asserts it.

### The it.21 claim is refuted by the commit 28 seconds before it

`ecbedf7` states the range "stopped at it.20 with the it.21 `n=8,192` wave killed
before its first cell landed." Both halves fail.

The cell landed and is journalled:

    {"t": "cell", "task": "e3_t8", "t_star": 8, "n_train": 8192, "seed": 0,
     "threads": 12, "steps": 150, "train_nrmse": 0.9732971180581712,
     "eval_nrmse": 1.0252097125261737, "boot_lo": 1.0168885291722383,
     "boot_hi": 1.0339123632622422, "verdict": "NO READING", "secs": 97.3}

— `results/r10_r10_it21_t8_n8192_capacity_softmax_t8.jsonl:7`, header `"when":
"2026-08-31 00:38:13"`, mirrored at `results/r10_it21_t8_n8192.log:4`. The wave
then self-truncated on budget at `:8`: `{"t": "dropped", ..., "steps": [600, 2400,
9600], "why": "unaffordable, see priced DAG"}`.

Both files were **committed by `67162a6` at 00:40:12**, 28 seconds before
`ecbedf7` at 00:40:40 said the cell never landed. Same author, same session,
consecutive commits.

A second, earlier `t*=8, n=8192` cell already existed at `eval_nrmse = 1.025127`
in `results/r10_it8_capacity_softmax_t8.jsonl`, so two readings of that geometry
were on record when the claim was written.

What is accurate in the message: no `R10_ITERATION_21.md` exists, so the *record*
series does stop at it.20. The *measurement* series does not.

### An orphaned process outlived the loop and is still running

`ps -ef` at the time of this read shows **PID 30409, `bash /tmp/waveB.sh`, started
2026-08-30 20:40:20, still live** — alongside PID 30359, a `tail -f` on the three
waveAC logs. `results/r10_it8_waveB.log` grew from 41,958 B to 42,084 B to
42,210 B across three samples inside this read.

The mechanism is one missing binary. `/tmp/waveB.sh:5` computes its drain
condition through `bc`:

    n_wrote=$(grep -hc 'WROTE' results/r10_it8_waveAC_t*.log 2>/dev/null | paste -sd+ | bc)

`bc` is not installed in this Git Bash, so `n_wrote` is empty, `:6` prints
`waveAC WROTE lines:  / 6`, and `:7`'s `[ "$n_wrote" -ge 6 ]` raises `integer
expression expected` instead of evaluating — the break condition can never be
reached and `:8` sleeps 45 s forever. `/tmp/waveB2.sh` (written 20:50) is the
repaired variant, using `wc -l | tr -d ' '` at `:3` and a `${n:-0}` default at
`:5`; it ran to completion and produced lines 1-26 of the log, including the three
real `n=32768` cells and `WAVE B COMPLETE`. The broken original was never killed
and appends from line 27 to the end: **319 `bc: command not found` triples** at
45 s each ≈ **3.99 h**, matching the wall-clock from waveB2's truncating open at
~20:50 to the sample above.

This is the same fail-open shape MISTAKES.md files as `V-16` (`:828-853`) — an
unreadable measurement not distinguished from a measured one — here in a shell
guard rather than a Python probe.

### Timeline

| time (IST) | event | evidence |
|---|---|---|
| 2026-08-30 17:03:41 | loop starts, iteration 1 | `started_at: "2026-08-30T11:33:41Z"`, `.claude/ralph-loop.R10.stopped.md:7` |
| 2026-08-30 19:34 | `bf2a769`, last commit before a 4 h 59 m gap | `git log` |
| 2026-08-30 20:40:20 | `/tmp/waveB.sh` launched, `bc` absent | `ps -ef` PID 30409 |
| 2026-08-30 20:50 | `/tmp/waveB2.sh` written, repaired | file mtime |
| 2026-08-30 22:30 – 2026-08-31 00:02 | four `tests/loop` suite snapshots recorded | `results/r10_loop_suite{,2,3,4}.txt` |
| 2026-08-31 00:33:51 | `0d23452` flushes the backlog: 12 iteration-record files, `MISTAKES.md +141`, `R10_MECHANISM.md +651` | `git show --stat 0d23452` |
| 2026-08-31 00:38:13 | it.21 `n=8192` cell journalled | `r10_r10_it21_t8_n8192_capacity_softmax_t8.jsonl:4,7` |
| 2026-08-31 00:40:12 | `67162a6` **commits** that cell | `git show --stat 67162a6` |
| 2026-08-31 00:40:40 | `ecbedf7` says the cell never landed | commit body |
| 2026-08-31 00:42:12 | state file renamed to `ralph-loop.R10.stopped.md` | file mtime |
| 2026-08-31 00:42 | session `8df7537d` transcript last write | `~/.claude/projects/…/8df7537d-….jsonl` mtime |
| ongoing | PID 30409 still spinning | `ps -ef`, log growth |

The 4 h 59 m commit gap and the 7-minute burst that closed it are worth stating
together: iterations 13 through 20 were written to disk between 22:15 and 00:27
(file mtimes) and all landed in one commit at 00:33:51. The git history does not
resolve the round's last third.

---

## Standing failures

### The set exists, it holds exactly 15, and it is not a prose ledger

The contract phrase points at `results/r10_loop_suite4.txt:21-35` — the
`short test summary info` block of a `tests/loop` run recorded 2026-08-31
00:02:24. Line 36 reads verbatim:

    15 failed, 499 passed, 3 warnings in 98.15s (0:01:38)

The only prose that names it as a set is `workdonenew.md:57`: *"499 passing in
`tests/loop`, **15 failing** — and every failure is a bound finding with a stated
route, not breakage."* The `499 / 15` pair matches the artifact exactly, and both
files entered the tree in the same commit, `0d23452`. `LOOP_PROMPT.md:176` calls
them *"the standing failing tests, which the fellows do not touch"*.

**The real count is 15, and 13 of the 15 are still standing.** The two that are
not are re-dated below.

The literal phrase "loop failure" returns zero hits repo-wide, as does "15
standing", "fifteen standing", "standing failure", "forensic read", "mount
request" and "deactivation commit". The set is 15 pytest node ids with no dates,
no status markers and no routes attached; the routes exist, but scattered across
`R10_GUARD_REPAIRS.md` and the test docstrings themselves.

### The 15, filed, dated and routed

Dates are the earliest suite snapshot in which each entry appears, `results/`
mtimes, IST.

| # | failure as filed | first seen | route |
|---|---|---|---|
| 1-10 | `test_conftest_import_is_order_dependent.py::test_no_test_file_imports_conftest_as_a_bare_module[…]` — ten parametrised cases: `tests/chase/`{`test_ceq_hub_package`, `test_hf_shipping`, `test_hub_package_hardening`, `test_kernel_contracts`, `test_rollback_flex_attention`, `test_schedule_rebuild`, `test_stochastic_P`}`.py` and `attic/tests/chase/`{`test_multizoom_cost`, `test_multizoom_kernel`, `test_multizoom_r5`}`.py` (`r10_loop_suite4.txt:21-30`) | roster of 5 at 08-30 22:30:06 (`r10_loop_suite.txt:21-25`), corrected to 10 at 22:46:59 (`r10_loop_suite2.txt:21-30`) | **STANDING, deferred by decision.** `R10_GUARD_REPAIRS.md:132`: *"repair priced (1 module, 11 sites, 6 function-local) and deferred: verification needs a full-suite run on a quiet tree"*. The defect is a bare `from conftest import …` resolved through `sys.path` (`tests/loop/test_conftest_import_is_order_dependent.py:3-16`), invisible on a full-tree run and surfacing only on subset invocations |
| 11 | `test_corpus_is_recoverable_and_verifiable.py::test_some_countable_unit_of_the_corpus_equals_the_readme_figure` (`:31`) | 08-30 22:30:06 | **STANDING, blocked upstream.** `R10_GUARD_REPAIRS.md:133`: `data/README.md` says 20,000 lines, the file has 211,766, *"no countable unit equals 20,000… Editing the number would fabricate provenance"* |
| 12 | `test_corpus_is_recoverable_and_verifiable.py::test_the_readme_derivation_matches_the_corpus_it_describes` (`:32`) | 08-30 22:30:06 | **STANDING**, same block. Measured 211,765 lines, 105,109 blank, zero `<|endoftext|>` markers; 10.6× the stated size (`tests/loop/test_corpus_is_recoverable_and_verifiable.py:31-40`) |
| 13 | `test_every_boundary_node_can_propagate.py::test_no_corpus_instance_has_a_boundary_node_that_cannot_propagate[reproduce-n1024-d4-t0.95]` (`:33`) | **08-31 00:02:24 — new in suite4, absent from all three earlier snapshots** | **STANDING, route stated in-file, not in any disposition document.** `tests/loop/test_every_boundary_node_can_propagate.py:33-42`: Amendment (B) to `admissible`, *"every b in B has at least one neighbour not in B"*, deferred because the corpus would need regenerating under twelve already-published rungs. Do()-bit moves the label by exactly 0.0 in every norm at `|B| = 80` |
| 14 | `test_manifest_refuses_an_absence_it_has_not_earned.py::test_no_weight_record_omits_a_declared_identity_field` (`:34`) | 08-30 22:30:06 | **STANDING by contract decision, half-repaired.** `manifest()` now returns `absent: ['device']` outside the hash inputs, 15 of 16 covered, hash `210b72c2f070d334`; rewriting the shipped `.pt` files refused under L-G2 (`R10_GUARD_REPAIRS.md:107-112`). `device` is absent from 60/60 weight records (`tests/loop/test_manifest_refuses_an_absence_it_has_not_earned.py:12-16`) |
| 15 | `test_the_bar_control_is_scored_out_of_sample.py::test_the_bar_control_is_not_scored_on_the_tensor_it_trained_on` (`:35`) | **08-30 23:43:39 — new in suite3** | **STANDING, and it bears on a live verdict.** `calibrate_bar` trains the two-feature control on `feats` (`negation_scope.py:1529`) and scores it on the same `feats` (`:1536`), while every arm it gates is scored out-of-sample at `seed + 12345`. Route: one line, score the control on a fresh draw; every block re-certified (`tests/loop/test_the_bar_control_is_scored_out_of_sample.py:39-41`) |

### Re-dating: the disposition document went stale within an hour

`R10_GUARD_REPAIRS.md` (mtime 2026-08-30 22:45) lists four items under *"What
stays RED, and why"* (`:130-136`). One of them — *"variance law `:367`"* — was
already green by the next snapshot: `test_the_variance_law_is_stated_once` fails
in `r10_loop_suite.txt:35` (22:30) and `r10_loop_suite2.txt:35` (22:46) and is
absent from `r10_loop_suite3.txt` (23:43) and `r10_loop_suite4.txt`. The table was
accurate when written and describes a RED that no longer exists.

Conversely, two of the current 15 — entries 13 and 15 — appeared **after** that
table was written and are routed nowhere in it. Of the 15 standing failures, the
project's own disposition document covers 12 and mis-states one it does cover.

### Resolved since, with what resolved them

Seven node ids that were standing at 22:30 or 22:46 are green by 00:02. They are
listed because the "15" is a count of a moving set, not of a fixed one.

| resolved failure | last seen RED | resolved by |
|---|---|---|
| `test_identity_manifest_covers_every_beyond_key_field::…[kind]` | `r10_loop_suite.txt:28` | `R10_GUARD_REPAIRS.md:118-124` — two parametrize entries added, *"both pass, so the refusal worked all along and only the coverage was missing"* |
| `…[torch_version]` | `r10_loop_suite.txt:29` | same |
| `test_manifest_refuses_an_absence_it_has_not_earned::test_the_published_cell_carries_every_declared_identity_field` | `r10_loop_suite.txt:30` | `R10_GUARD_REPAIRS.md:107-112` |
| `test_no_module_writes_a_file_at_import[scale/it13_mars_binding.py]` | `r10_loop_suite.txt:32` | green by `r10_loop_suite2.txt` (22:46); no document names the repair |
| `test_no_module_writes_a_file_at_import[scale/p1prime.py]` | `r10_loop_suite.txt:33` | same |
| `test_no_module_writes_a_file_at_import[scale/replay_census.py]` | `r10_loop_suite.txt:34` | same |
| `test_phase1a_modules_are_bound[scale/it11_verdict.py]` | `r10_loop_suite2.txt:34` | green by `r10_loop_suite3.txt` (23:43); no document names the repair |
| `test_the_variance_law_is_stated_once::test_the_variance_law_is_stated_consistently` | `r10_loop_suite2.txt:35` | green by `r10_loop_suite3.txt`; `R10_GUARD_REPAIRS.md:135` still lists it as RED |

### The count is stable and the membership is not

| snapshot | time | failed | passed |
|---|---|---|---|
| `r10_loop_suite.txt:35` | 08-30 22:30:06 | 15 | 465 |
| `r10_loop_suite2.txt:35` | 08-30 22:46:59 | 15 | 471 |
| `r10_loop_suite3.txt:34` | 08-30 23:43:39 | **14** | 479 |
| `r10_loop_suite4.txt:36` | 08-31 00:02:24 | 15 | 499 |

Three of the four snapshots read 15 with **three different membership sets**.
Between the first and the last, 34 tests moved into passing and the failure set
turned over by more than half. Any contract clause that treats "the 15" as a fixed
enumerable set is treating a coincidence of three counts as an identity. No file
in the repository cites `r10_loop_suite*.txt` by name.

### Rule D-3 is not in this repository

The rule quoted in the mount contract — deactivation commit, MISTAKES.md, the 15
standing loop failures, iteration count from the DAG's critical path — exists
nowhere in the tree. The only `D-3`s present are `MISTAKES.md:727` (*"A difficulty
dial that does not vary"*, one of five `D-` design-level failures at `:677`,
`:710`, `:727`, `:740`, `:755`) and `results/r10_inspector_phase1a.md:141`
(*"it11_verdict.demo() passes on an empty journal set"*). Neither mentions loops,
mount requests or forensic reads. `PREREGISTRATION_HOLE_AUDIT.md:259` cites
"§2, D-3" and resolves to the second of these.

**The contract asserts a rule that is not in the repository.** The set it points
at is real and the count is right; the rule text itself is not on disk and cannot
be checked against a source. Recorded rather than reconciled.

---

## Mistake mechanisms

`MISTAKES.md` holds **41 entries** across four declared classes: V-1 … V-14,
V-14a, V-15 … V-18 (19, `:34`-`:891`), P-1 … P-8 (8, `:289`-`:387`), M-1 … M-9
(9, `:427`-`:580`), D-1 … D-5 (5, `:677`-`:755`). Note that V-14a and V-15 … V-18
sit *after* the D section (`:770`-`:926`), appended out of the file's own order.

Grouped by failure mechanism rather than by that classification, each entry
assigned once, counts summing to 41:

| # | mechanism | count | entries |
|---|---|---|---|
| 1 | **The check has no rejection region** — the comparison cannot come out false, by arithmetic, by construction, or because the statistic cannot reach the threshold | **9** | V-2, V-3, V-5, V-8, V-10, V-11, V-12, M-5, M-9 |
| 2 | **The instrument never reached its subject, and empty reach was read as a finding about the world** | **6** | V-6, V-7, V-13, V-14, V-14a, P-4 |
| 3 | **A number severed from a live producer, or a cross-reference pointing at nothing** | **6** | P-1, P-2, P-3, P-5, P-6, P-7 |
| 4 | **Arm and control differ in a way nobody asserted, so the contrast is void before the data** | **6** | V-1, M-1, M-4, D-1, D-2, D-5 |
| 5 | **A rate, threshold or bound carried outside the geometry that produced it** | **5** (4 instances + M-2, filed as the model done right) | V-17, P-8, M-3, M-8, M-2 |
| 6 | **A partial, degenerate or absent state rendered as a verdict** | **5** | V-4, V-9, M-6, M-7, D-4 |
| 7 | **A proxy standing in for the property the rule names** | **4** | V-15, V-16, V-18, D-3 |

Mechanism 7 is small in `MISTAKES.md` and enormous elsewhere: `R10_MECHANISM.md`
is a dedicated ledger of that one mechanism carrying **24 instances numbered to
25** (`:229`), with no instance 15 and the gap deliberately left open (`:227`).
Counted across both files it is the largest single mechanism in the repository by
a wide margin. Its own title has been wrong three times, most instructively when a
derived count scanned the wrong region and published 22 for a file holding 20
(`R10_MECHANISM.md:231`).

### Sharpest example per mechanism

**1 — no rejection region.** `M-9` (`MISTAKES.md:580-592`): at N = 5 seeds the
verdict is a sign test, and *"the finest achievable two-sided p at N = 5 is
`0.0625`, not the `0.05` the project quotes. The design cannot produce the
significance level it reports, whatever the data say."* This covers every 5-seed
reading in the repository including the standing `+0.108437` headline. Sharper
than `V-3` because it invalidates the flagship result rather than one control.

**2 — empty reach.** `V-14` (`MISTAKES.md:251-258`): `chase_struck_coverage.py`
tested its exclusion list against the absolute path, every worktree lives under
`<repo>/.claude/worktrees/`, so `.claude` was a component of every file's absolute
path. *"366 candidate files became 0."* The tool printed `SCANNING 0 PATHS THE
SHIPPED CHECK DOES NOT COVER`, `uncovered .md: 0, uncovered .py: 0`, and exited 0
— while shipping a must-fire control that passed throughout, because the plant
entered at the matcher and never traversed the selection stage.

**3 — no live producer.** `P-2` (`MISTAKES.md:300-314`): the IMPACT gate figures
(truncation k2 `0.3243`, decoder local `1.0039` vs planted `0.06738`, sign gate
degrade `0.8793` CI `[0.8300, 0.9145]`) live in commit `74e5590`'s body and in
`DONE.md`. *"Nothing on disk reproduces them"* — and both `impact` and
`impact_hetero` are registered in `M3_TASKS` with the four gates their own
docstring requires never executed.

**4 — unasserted arm/control difference.** `D-1` (`MISTAKES.md:677-708`), which
the file itself calls *"the largest one in the repository"* and says *"subsumes
most of the null results"*: every label is a scalar point prediction at position
`s-1` (`scale/m3_quintuple.py:483`), which is exactly the single-location regime
where one softmax layer is provably Bayes-optimal. The escape exists
(`vector_readout`, `scale/m3_quintuple.py:368`) and defaults to `False`.

**5 — bound carried out of its geometry.** `P-8` (`MISTAKES.md:387-407`):
`results/r9_systems_gate.md:196` priced a lane at `≈ 29.6 h`; measured, `5.9 h` —
a 5× miss. What makes it the sharpest is that the correct caveat was already
written, in the right place, at `:328`. *"A caveat below a number does not travel
with the number; the number travels alone."*

**6 — partial state as verdict.** `M-6` (`MISTAKES.md:518-527`):
`foreman_looped.py`'s own `falsifier()` returns `complete: False` and refuses a
verdict, while the table holds 2 of 6 cells and the CUDA lane 22 of 60 — and
`STATE.md:21` still reads "RUN IN FLIGHT" for a lane that is complete.

**7 — proxy for the property.** `V-18` (`MISTAKES.md:891-909`): `it11_verdict.py`
deduplicates journal rows and refuses below N=8; an ad-hoc analysis written *"in
the same round, by the same author, one iteration after the module shipped"*
regrouped inline and read means over rows, putting the crossing at 12,789 against
the seed-correct 12,780. *"The magnitude of the error is set by how duplicated the
journal happens to be, not by anything the reader controls."*

That duplication is present and checkable right now:
`results/r10_it8_capacity_softmax_t8.jsonl` carries `n=32768 seed=0` twice at
`0.972372` and `n=32768 seed=4` twice at `0.978695` — bit-identical rows, exactly
the condition `V-18` describes.

### Exposure of the staged work

The staged work — a state-prediction head with a fan-out detector, barrier
regression, and a from-scratch local training run on one RTX 4060 — has **no
footprint in this tree**. `git status --porcelain -uall` reports one modified log
and nothing else; `git stash list` holds one unrelated R9 entry; grep for
`fan-out`, `fanout`, `barrier`, `state-prediction`, `next-state` across all `.md`
and `.py` outside `attic/` and `.claude/worktrees/` returns zero hits. The
exposures below are therefore structural, derived from the shape of the work
rather than from code.

**Mechanism 1 — the fan-out detector is a degenerate-label risk.** A detector over
graph structure inherits `V-8` exactly: if the corpus's fan-out distribution is
one-sided, the PASS half's label is constant, NRMSE normalises by `std(y)` and
returns `nan`, and `float('nan') >= 1.0` is `False` in Python — so a bare
threshold passes it silently (`MISTAKES.md:149-152`, `scale/m3_capability.py`
`bad()`). Requirement before the first reading: fan-out `sd > 0`, `0 < frac < 1`,
both classes non-empty, discard count printed, per `BOARD.md:269`.

**Mechanism 1 and 7 — barrier regression is a threshold with unstated units.**
`V-17` (`MISTAKES.md:855-889`) is the precedent and it is recent: `delta = 0.5`
imported from the chain task's `flipper_dependence` clause onto a harmonic corpus
made **ten of twelve rungs fail by construction**, because `fd_max × |B|` is
near-constant at 3.741 and the corpus runs `|B| = 4..80`. The test is one
question: what value does the barrier take under a null or trivial predictor? An
anchored quantity travels (NRMSE's 1.0); a ratio calibrated against a generator
does not.

**Mechanism 4 — the state-prediction head lands directly on D-1's escape, and on
its trap.** Predicting the next state is a vector-valued label, which is precisely
the shape `D-1` says leaves softmax's proven-optimal regime — the one change that
makes the comparison worth running. But `MISTAKES.md:704-708` states the
condition: it must go in a separate lane with its own journal and weights
directory, because changing `forward` voids the `PUBLISHED_SOFTMAX_8192`
reproduction gate at `scale/m3_quintuple.py:696-708`. Racing the new head against
a softmax baseline still on the scalar readout reproduces `D-1` with the arms
swapped.

**Mechanism 5 — a local 4060 run is where M-3 and M-8 have already bitten twice.**
The costing law is written and is binding
(`PHASE2_CONTRACT_V_MAIN_4.md:190-198`): estimates are a count of 600-step
run-equivalents read off the source that will execute them, times a rate
re-measured at the moment of spend — **20.58 s/RE uncontended, 48.89 s/RE at four
concurrent seats**. Two measured facts a parameter-count model is blind to, at
equal parameter count (`:200-206`): the operator costs **3.3× (heads) and 4.7×
(seq)** its own control in wall-clock, and `MODEL_CARD.md`'s shipped 1.32× is a
seq-128 / H-4 / 3.3M number that does not transfer. The card is 8.0 GiB
(`README.md:480`). The it.21 wave that ended this loop dropped steps `[600, 2400,
9600]` as unaffordable on exactly this hardware.

**Mechanism 2 — any tree-wide scan written for the new head repeats V-13/V-14
unless scoped to the tracked set.** `git worktree list` currently reports **ten
entries**, nine of them full nested copies under `.claude/worktrees/`. The rule is
`MISTAKES.md:230-235`: scope to `git ls-files`, never to a directory blocklist,
and state the expected hit count before running.

**Mechanism 6 and 7 — training journals must route through the shared reader.**
`V-18` and the duplicate rows demonstrated above apply to any new reader over
training journals. The check is a grep for the grouping key across analysis
scripts, confirming each hit calls the deduplicating module rather than
regrouping inline (`MISTAKES.md:911-914`).

**One live verdict is already exposed.** Standing failure 15 says
`calibrate_bar`'s control is scored in-sample while every arm it gates is scored
out-of-sample. `0d23452`'s headline — *"the t\*=32 wall is a data budget, not an
architecture limit"* — rests on a `NOT LEARNABLE` reading at `t*=32` where the bar
printed `BAR CALIBRATED` and softmax read 1.0034 over eight seeds. The test's own
docstring states the consequence: that verdict *"is consistent with EITHER
'softmax is the limitation' or 'the bar overstated the task', and nothing in the
round currently separates them"*
(`tests/loop/test_the_bar_control_is_scored_out_of_sample.py:27-33`). Any new head
certified through the same bar inherits the ambiguity.

---

## Iteration count from the DAG's critical path

D-3 requires the number come from the DAG, not from an estimate.

The completion promise names two ranges: *"CEQ R10 COMPLETE: v-main.3M it.40 and
v-main.4 Phase 2 it.40 both done"*. They are **strictly sequential**, not
parallel. `PHASE2_CONTRACT_V_MAIN_4.md:22-30` fixes three entry preconditions —
S3 decided at v-main.3M it.30, S4 at it.33, S5 at it.35 — and states: *"Phase 2
cannot begin before those three verdicts exist. Starting it earlier freezes a
reference to an undecided architecture, and every kernel in it.4–it.6 is bound to
that reference."*

- v-main.3M remaining: it.21 → it.40 = **20** (it.21's first cell has landed; its
  remaining step budget was dropped as unaffordable).
- Phase 2: **40**, enumerated as 40 numbered rows at
  `PHASE2_CONTRACT_V_MAIN_4.md:126-166`.
- **Critical path from the present state to the promise: 60 iterations.** The
  promise as written names 80; 20 are done.

Measured turn cost, for a mount request that needs a budget rather than a count:
the state file records `iteration: 600` over a run from 2026-08-30 17:03:41 to
2026-08-31 00:42 — 458 minutes, **45.8 s per ralph turn** — which bought 20
completed research iterations plus one partial, i.e. **≈30 ralph turns per
research iteration**. Sixty more research iterations at that measured rate is
≈1,800 ralph turns. That rate is `INHERITED` under the phase's own L-TIME clause
(`PHASE2_CONTRACT_V_MAIN_4.md:219-220`) and must be re-measured by whichever
iteration quotes it.

---

## Limits

- The `tests/loop` suite was **not re-run**. This node is read-only and pytest
  writes `.pytest_cache/`. Every failure count, membership set and pass count here
  is read from the four recorded artifacts `results/r10_loop_suite{,2,3,4}.txt`,
  the newest of which is 2026-08-31 00:02:24 — before the round's last three
  commits. The current live count may differ from 15.
- The date on which `.claude/` entered `.git/info/exclude` is not recoverable:
  `info/exclude` is not versioned. It is bounded to the interval between
  `9e1390c` (2026-08-26 11:39:52, the last commit touching a loop file) and the
  R10 launch on 2026-08-30.
- `active: false` being inert is established against the plugin at
  `~/.claude/plugins/cache/claude-plugins-official/ralph-loop/1.0.0/`. If the
  session ran against the marketplace copy at
  `~/.claude/plugins/marketplaces/claude-plugins-official/plugins/ralph-loop/`,
  that copy was not diffed against the cache.
- PID 30409 was **left running**. Killing it is a write to process state outside
  this node's scope. It appends ~3 lines per 45 s to
  `results/r10_it8_waveB.log`, which is a tracked file, so the working tree will
  keep drifting until someone ends it. Its parent `tail -f` is PID 30359.
- Whether the manual kill at 00:42:12 was prompted by the unaffordable it.21
  budget, by the 4 h 59 m commit gap, or by an operator decision recorded nowhere
  on disk cannot be settled from artifacts. The temporal adjacency (00:39:10
  artifact write → 00:40:12 commit → 00:40:40 deactivation → 00:42:12 rename) is
  what supports the reading, not a stated reason.
- The mechanism taxonomy assigns each of the 41 entries a single primary
  mechanism. Several entries carry a second (`P-8` is both mechanism 5 and
  mechanism 3; `V-14` is both 2 and 7; `M-9` is both 1 and 3, which
  `MISTAKES.md:621` says explicitly). Counts would shift under multiple
  assignment; the totals above are one-per-entry and sum to 41 by construction.
- The staged work's exposure section is derived from the described shape of that
  work, since no file in the tree implements it. Once code exists, the six
  exposures should be re-checked against it rather than against the description.
