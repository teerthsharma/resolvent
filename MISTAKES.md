# MISTAKES — the failure taxonomy of this repository

Read this before writing a control, a gate, a bar, or a number that will end up
in a verdict. Every entry below is a mistake this repository actually made, with
the place it is written down. The entry that matters is the third line of each:
the **Rule**, which is what stops the type from recurring. The instances are
here so the rules are not abstract advice.

Four classes, in the order they cost the most:

- **V — vacuous controls.** A control that cannot fail. Fourteen were struck
  across five authors (`STATE.md:53`, `done7.md:175-179`), and the count is a
  floor, not a total: two later items were each independently labelled "the
  fifteenth" (`DONE.md:1228`, `PIVOT_EXCLUSION_FALSIFIER.md:50-52`). A vacuous
  control is worse than no control because it reads as evidence — one of them
  stood two iterations and was used to strike a colleague's evidence
  (`done7.md:187-188`).
- **P — provenance failures.** A number with no live producer, or a claim that
  was true once.
- **M — measurement failures.** The instrument measured something, but not the
  thing the verdict names.
- **D — design-level failures.** The experiment was built so that its answer was
  fixed before any data arrived.

A cross-cutting fact worth stating once: `M3_TASKS` is a plain dict literal and
**nothing validates an entry on import** (FINDINGS F). Gates are enforced by
hand-written tests or not at all. Most of class V entered through a registration
that nobody could have failed.

---

## V — Vacuous controls

### V-1. Two registry keys, one corpus

`scale/negation_scope.py` registered `impact_hetero` against
`make_impact_batch` — the same builder `impact` binds — so both keys returned a
byte-identical 4-tuple and a heterogeneous-plant arm drew the homogeneous
corpus. `make_impact_hetero_batch` existed (`scale/impact.py:601`) and was imported
(`scale/negation_scope.py:59`), never used. The arm was its own
baseline by construction: no draw could separate them.

Contrast the licit version of the same shape. `rag_multihop_t*` registers the
e3 corpus a second time under document-graph naming and says so
(`scale/negation_scope.py:907-921`): *"the rag reading is a NAMING of the same
corpus object, not a second corpus"*, and — the part that makes it honest —
*"IDENTICAL TENSORS IS TESTED, NOT STATED"*, asserted bitwise in
`tests/cameron/test_u1_rag_registration.py`. The difference between V-1 and U1
is not the identical tensors. It is that one entry claimed a different corpus
and the other claimed the same one.

**Rule.** Two registry keys claiming two corpora must be asserted to *differ* on
drawn instances, with the count reported, exactly as two keys claiming one
corpus must be asserted to *match*. Register the assertion in the same commit as
the entry. Bound here by
`tests/cameron/test_impact_hetero_is_not_its_own_baseline.py`.

### V-2. Hand-built minimal example where right and wrong coincide

A greedy must-fire used a hand-built 2×2 example and could not fire, *"because
on two points the monotone assignment is forced, so greedy and optimal coincide
for every such instance"* (`CHECKLIST.md:805`). The repair was drawn instances
with a count: greedy strictly worse in **111/400** random (4,9) instances. The
same line names the pattern: *"A hand-built minimal example is exactly where a
control goes vacuous, because the smallest case is usually where right and wrong
answers coincide."*

**Rule.** Draw the instance, never hand-build it. Report the count of draws on
which the control discriminates, and require that count to be bounded away from
both 0 and n — the repair above asserted `>50/400` as well as `never beats`.

### V-3. The assertion is an algebraic identity of your own construction

`pooled < tail` passed **400/400 strict and could not fail**, because pooling
minimises over a superset (`DONE.md:4775`, `CHECKLIST.md:889`). The same defect
sat in the bar itself: `calibrate_bar`'s original clauses 1 and 3 were
`nrmse(y.mean(), y)`, which is 1.0 by the definition of NRMSE, and
`nrmse(oracle(x,f,p), y)`, which is `nrmse(t, t)` because `make_batch` *returns*
`oracle(x,f,p)` as `y` (`scale/negation_scope.py:1133-1137`). A flipper-blind
label — the entire premise of the task removed — passed the gate and printed
BAR CALIBRATED.

**Rule.** Before asserting `A < B`, ask what set each side ranges over. If one
is a restriction of the other, or one is computed from the other, the assertion
is arithmetic and measures nothing. Keep such clauses only if you label them as
definitional in the output, the way `predict_the_mean` now is.

### V-4. Fired, but on the wrong cause

Row 0 of `tril(-1)` sums to exactly 0.0, giving NaN, which makes `torch.equal`
return False — so the control fired, for a reason unrelated to the property
under test (`CHECKLIST.md:851`).

**Rule.** A control that fires must be made to state *why*. Assert the
intermediate quantity, not just the final boolean, and check the fixture for
degenerate rows before reading a False as a finding.

### V-5. Slice-to-nothing

A genesis assertion sliced a digest to zero characters and asserted a label was
present in the result: `X in ""[:0] + X` reduces to `X in "" + X`, true for
every input (`CHECKLIST.md:1096`, `done7.md:181-183`).

**Rule.** Any assertion built from string or tensor slicing must be run once
against an input it should REJECT. If you cannot construct that input, the
assertion has no rejection region.

### V-6. The branch under test never ran

An overflow test read `left_cone=False, converged=True, steps=1` — *"settled
before it could drift, branch never ran"* (`CHECKLIST.md:1133`).

**Rule.** Instrument the branch. Assert that the code path under test executed —
a counter, a recorded step count, a sentinel — in the same test that asserts its
outcome. Green from a branch that did not run is indistinguishable from green.

### V-7. A search structurally incapable of finding anything, read as absence

`journal_scan` read only top-level keys of records that nest their payload under
`value`. It reported *"zero readings above 1.0"*; the truth was **22 of 68**. It
*"shipped, stood two iterations, and was used to strike a colleague's
evidence"* (`done7.md:186-188`). The repair made it a mechanism rather than a
discipline: the caller supplies a witness that must be recovered, and a scan
whose witness is missing **raises** instead of returning an empty list
(`done7.md:194-200`). That repair's own tests then found two bugs in it, the
worse being that the witness was checked against paths *walked* rather than
paths the *selector keeps* — so a transposed selector passed. As `done7.md:199`
puts it: a witness that cannot fail on a wrong selector is precisely the disease
it was written to cure.

**Rule.** A reported absence needs a planted positive on identical instances,
identical features, identical split. Zero results is a claim about the search
before it is a claim about the world, and it must be paid for with a positive
the search is required to find.

### V-8. The PASS half's label is constant

The fourteenth strike. `SupercriticalDense_S2Rips_256`: all 256 nodes in **one
component**, 0 isolated, base rate 1.000000, label **sd 0.0**, NRMSE undefined;
drew 1024 same / 0 different (`CHECKLIST.md:1247`). A one-component graph cannot
answer "same component?" any way but yes. It *"sat in the control half of a
must-fire"* and was relayed into a dispatch as *"that contrast IS the control"*
without checking the pass-case could produce a non-constant label
(`CHECKLIST.md:1248`).

**Rule.** Check the PASS half's label is non-degenerate before the FAIL half is
interesting: `sd > 0`, `0 < frac < 1`, both classes non-empty, discard count
reported. `BOARD.md:269` now requires every zero to be paired with exactly that
triple. NRMSE is normalised by `std(y)`, so a constant label does not produce a
bad number — it produces `nan`, and `float('nan') >= 1.0` is False in Python, so
a bare threshold lets it through as a silent pass (`scale/m3_capability.py`
`bad()`).

### V-9. A repair that changes nothing

`exclude=(0,)` writes `-inf` into a top-k that had selected `s-1` in **0 of 32**
draws, so the exclusion removed a candidate that was never chosen.
`PIVOT_EXCLUSION_FALSIFIER.md:50-52`: *"Shipping it as the fix would be a change
that moves no number, reported as a fix."* `STATE.md:194` states the type
exactly: *"a vacuous control shipped as the repair for a vacuous-control
problem."*

**Rule.** A repair must be shown to change the object it repairs, on the live
object, with the before and after both printed. Delete the repair in-process and
require the number to move. If the two readings agree, the repair is decoration
regardless of how correct the reasoning behind it was.

### V-10. A gate whose threshold is satisfied by construction

`κ_emp ≤ κ_cert` passed **30/30 because `0.1 ≤ 1.0`** (`DONE.md:5535`,
`CHECKLIST.md:773`). The gate compared a quantity that lives near 0.1 against a
bound that sits at 1.0.

**Rule.** Compute the control's expected value before you run it. If the
comparison holds at the extremes of the quantity's own range, the gate has no
rejection region and is vacuous before it executes.

### V-11. A precondition satisfied at every real draw

An admissibility rule read *"admissible where the third mass is below ~0.115"*
while characterising masses `0.02–0.2` against a **measured median of
`1.2454e-20`** — *"vacuously satisfied at every real draw"* (`CHECKLIST.md:989`).

**Rule.** State a precondition against the measured distribution of the quantity
it constrains, not against its theoretical range. Print the median and the
fraction of draws the precondition excludes; a precondition that excludes 0% is
not a precondition.

### V-12. A single absorbing target makes the label constant

*"A single absorbing target makes the label constant to `1.11e-14`"*
(`DONE.md:1228`) — caught pre-dispatch and labelled the fifteenth vacuous
control.

**Rule.** Trace the label back to the structure that generates it. If one
structural choice collapses the label's support, the task's variance is a
property of that choice and not of the corpus. Print label sd at construction
time, in the builder, not in the reading.

---

### V-13. A search whose walk includes nested checkouts

The sign-flip of V-7, found by the same discipline that V-7's rule installed.
`tests/deimos/test_deimos_r9_iteration1.py` asked "does `return_residual` appear
anywhere outside its own definition" with `root.rglob("*.py")`, excluding only
`.git` and `node_modules`. This repository keeps every agent's worktree nested
under `.claude/worktrees/`, each a **full copy of the tree** — nine of them as
this is written (`git worktree list` shows ten entries, the tenth being the
primary checkout). The walk therefore descended into every sibling worktree's own
`ceq/nash.py`. Deimos measured **12 hits — 10 phantom copies, 1 the test file
quoting the symbol, 1 the real file.** Re-counting today gives nine phantom
copies, not ten, which is the second half of the defect: **the wrong number is
not even stable**, because it tracks how many agents happen to hold a worktree
at the moment the search runs.

The finding being made is an *absence* claim — "nothing reads this residual" —
so an over-finding search inverts it into a false *presence*. The search
returned twelve files against a tracked truth of one, and **ten of the twelve
were copies of the very file being asked about**, so the evidence that looked
like "this symbol is used all over the tree" was the definition counted eleven
times. Deimos's own line for it, now in the fixed test: *"A search that
over-finds is the same failure as one that under-finds, sign flipped."*

The repaired test scopes to `git ls-files -- '*.py'` and skips itself
(`tests/deimos/test_deimos_r9_iteration1.py:265-291`), and asserts the hit list
equals exactly `[ceq/nash.py]` — so a future reader that starts consuming the
residual fails the test rather than quietly making the finding stale.

**Rule.** Scope a repo-wide search to the **tracked set** (`git ls-files`), never
to a directory blocklist. A blocklist has to be extended every time a new nesting
appears on disk and is silently wrong until someone notices; the tracked set
never needs extending. And state the search's expected hit count before running
it — V-7 and V-13 are the same defect at opposite signs, so *both* directions
need the count, not just the zero.


### V-14. The control that validates the matcher and never the reach

The one that survives rule 5 being obeyed, which is what makes it a new type
rather than another instance of V-7.

`scale/chase_struck_coverage.py` scans the tree for struck constants asserted
without their strike marker. It **ships a must-fire control**, it runs it before
anything else, and `main` refuses to proceed if the control does not fire
(`scale/chase_struck_coverage.py:120-122`). The control plants a struck number in a paragraph and requires one
hit, then plants the same number carrying a strike marker and requires zero — a
planted positive AND a non-degenerate negative half, exactly what V-7's rule
asks for.

It passed throughout while the scanner reached **nothing**. The exclusion list
was tested against `p.parts`, the ABSOLUTE path, and every worktree in this
project lives under `<repo>/.claude/worktrees/<name>/`, so `.claude` was a
component of every file's absolute path and the filter dropped all of them.
**366 candidate files became 0.** The tool printed `SCANNING 0 PATHS THE SHIPPED
CHECK DOES NOT COVER` and `uncovered .md: 0, uncovered .py: 0` and exited 0 —
which reads as "no struck constant is asserted anywhere uncovered" and is in
fact "no path was looked at".

**The mechanism, and it is one line.** `control()` calls `scan_text(...)` on a
literal string (`scale/chase_struck_coverage.py:107-110`). `collect_targets()` is never on that path. So the
control exercised the **matcher** and never the **reach**, and the two halves of
the instrument fail independently. Rule 5 said "a planted positive on identical
instances"; the planting was on identical *text* and not on an identical *path
through the instrument*, and the whole defect lived in the segment the plant
skipped.

It was caught sideways, and that is worth recording too: adding three constants
to `STRUCK` (9 → 12) produced no collateral at all, and Venus treated a
convenient result as suspicious rather than as good news.

**Rule.** A planted positive must enter at the instrument's **front door** and
traverse every stage the real input traverses — for a scanner, that means
writing the plant to a file inside the scan root and running the whole pipeline,
not calling the matcher. State which stages the plant passes through; any stage
it skips is unmeasured, and a control that skips the selection stage cannot see
a selector that selects nothing. The paired check is cheap and belongs beside
it: **assert the target list is non-empty and bounded**, because "how many
things did I look at" is a different question from "would I recognise one".

Note for the taxonomy chapter: this is a **structure** comparison (did the walk
reach the right set?) guarded by a **value** comparison (does the matcher
recognise the right text?). The value comparison was correct and stayed correct.
That pairing — a value control standing in for a structure claim — is where this
class of defect keeps coming from.

## P — Provenance failures

### P-1. A number with no live producer

`5.4944e-13` existed only in a code comment and in prose. The real probe reads
`7.307e-13`–`8.405e-13` at `--tol 1e-15 --steps 400` (`CHECKLIST.md:665`), and
building the registry to catch it *"then caught two more unmarked assertions I
had missed"* (same line).

**Rule.** Every number in a comment, docstring or document names the command
that reproduces it. A number whose producer cannot be run is prose, and prose
does not enter a verdict.

### P-2. A number whose only home is a commit message

The IMPACT gate figures — truncation k2 `0.3243`, decoder local `1.0039` vs
planted `0.06738`, sign gate degrade `0.8793` CI `[0.8300, 0.9145]` — appear in
commit `74e5590`'s body and in `DONE.md`. **Nothing on disk reproduces them**:
no results file, no test, no CLI (FINDINGS B5). The module's own docstring
forbade the registration that happened anyway: `scale/impact.py:23` says
*"Register as IMPACT in M3_TASKS only past all four gates + planted controls"*,
and both `impact` and `impact_hetero` are registered in `M3_TASKS` with those
four gates never executed
(FINDINGS A4).

**Rule.** A gate that has not run has not passed. Registration is gated on an
artefact in `results/` or a test that executes the gate, never on a narrative
that the gate was run. A commit message is not a producer.

### P-3. A stale claim never retracted

`LOOP_PROMPT.md:53-55` calls iterating the representation *"the one live
direction that has not yet been measured here"* while
`scale/foreman_looped.py` is a complete 305-line instrument with a bind battery
and an in-file pre-registration, and `results/foreman_looped.jsonl` holds 2 of 6
required cells (FINDINGS B1). `PIVOT_EXCLUSION_FALSIFIER.md:156` says
`settled_plus` *"was still running when this was written"*; it finished, and
`results/etask_k5e_plus.txt` records `settled_plus 0.956787` — also worse. The
document was never updated (FINDINGS B3).

**Rule.** A claim of the form "not yet measured" or "still running" carries the
date it was written and is checked against `results/` before it is read. When a
run finishes, the document that predicted it is edited in the same commit that
lands the number.

### P-4. Claimed scaffolding that does not exist

`BOARD.md:288` says *"U2 mujoco contact-graph DSU n≥1024 and U3 Tonnetz lattice
graphs — scaffolding code exists as stubs."* `git grep -i tonnetz` hits
`BOARD.md` twice and `DONE.md` once. **Zero `.py` anywhere.** No U2
contact-graph builder exists (FINDINGS B2).

**Rule.** An existence claim about this codebase is a `GUESS` until grepped.
This is where invented paths and flags come from. Paste the grep, with its hit
count, beside the claim.

### P-5. Doc rot pointing at nothing

`scale/arm_s.py:56-58` names `gram_audit`, which does not exist anywhere in this
repository. **The retraction is already in place** and is the model for this
class: the sentence now reads *"no function or test named `gram_audit` exists in
this repository, and an earlier revision of this sentence claimed one did (doc
rot, recorded as STATE.md item 39)"*, and it goes on to name which real
artefacts are the nearest analogies and what they do not cover. (FINDINGS B4
describes this file as citing `gram_audit` with the rot never fixed; the claim
that the file asserts the symbol exists is **wrong as of this commit** — what
remains true is that no measurement of pivot-coordinate underflow is on record.)
`scale/capability_table.py`'s `TASK_SOURCE` string says `m3_quintuple.py` *"has
no `--task` flag"*; `--task` was added at `scale/m3_quintuple.py:824` (FINDINGS
B8). The same `render()` still emits *"This package carries NO trained
weights"*, and `write_artifact` (`:443`) would clobber the weight-manifest sync
from commit `0162bdd` on the next table cut.

**Rule.** A cross-reference is dead weight unless something fails when it
breaks. Assert cited symbols exist — one test that imports every name a
docstring names is cheaper than the audit that finds them missing.

### P-6. Line-reference drift

`scale/m3_quintuple.py:335` cites `arm_s.pivots_of (line 97)`; it is at `106`.
`scale/m3_flops.py:44-84` cites nine stale lines.
`M3_QUINTUPLE_PREREGISTERED_READING.md:38` cites `:236`; it is `256-257`
(FINDINGS B7).

**Rule.** Cite the symbol, not the line: `arm_s.pivots_of`, not
`arm_s.py:97`. Where a line number is genuinely needed, it is quoted with the
commit it was read at. **Do not trust in-tree docstring line numbers.**

### P-7. Vocabulary with no referent

`§U`, `X₂₁`, `X₂₂`, "contract v10.1", "amendment v10.4" appear across
`BOARD.md:282,284,307` and `DONE.md`. `LOOP_PROMPT.md` has 17 headings and
**none is §U**; no R10 plan or criteria exists (FINDINGS B6).

**Rule.** A section reference names a heading that can be grepped in the
document it claims to be in. Coined identifiers get defined once, in one file,
before they are used in a second.

---

### P-8. A headline that states an upper bound as a price

`results/r9_systems_gate.md:196` priced the per-row lane at **`≈ 29.6 h`** for
ten units. The estimate assumed the arm calls `_alpha` once per query row, so
every term carried a factor of `s`. The arm that was actually built shares the
pivot set, the Gram and `A_P @ V` across rows and runs **one** Python loop over
an `[n·s, k]` tensor instead of `s` loops over `[n, k]`, so the iteration count
does not scale with `s` at all. Measured at the gate's own geometry: **`5.9 h`**
— a **5×** miss, superseded in place in the same file.

**What makes this an entry rather than an arithmetic slip is that the caveat was
already there.** The gate's own limits paragraph carried the clause that
predicted the failure, in the right section and correctly worded —
`results/r9_systems_gate.md:328`: *"an implementation that batches the query
rows differently could beat it."* The same paragraph now records what happened
next (`:332-334`): *"The caveat was correct, was written in the right place, and
was still not enough: a derivation carrying a live 'an implementation could beat
this' clause is an upper bound and should have been labelled one in §3.2's own
table rather than only in this paragraph."* The controller read the headline and
ruled a pilot-only strategy on it. **A caveat below a number does not travel
with the number; the number travels alone.**

Two different corrections of this one headline are on record and they are not
the same correction. Neptune's `5.9 h` is a **measurement of the arm that got
built**. Mercury's `15.5 h` (`results/r9_pricing.md`, and see M-8) is the **same
estimate re-priced at correct per-arm rates**, with the implementation
assumption left standing. Both are right about different questions, and quoting
either as "the corrected price" without saying which question it answers repeats
the original defect one level up.

**Rule.** A bound carries its direction **in the headline**, not in a limits
paragraph beneath it: write `≤ 29.6 h (naive upper bound, assumes per-row
_alpha)`, never `≈ 29.6 h`. If the headline number and the caveat get separated
— and they always do, because a headline is what gets quoted — the caveat was
decoration. When a measurement is affordable, a derivation from an assumed
implementation is not a price at all.


## M — Measurement failures

### M-1. Train and eval saw different preprocessing

`calibrate_bar` trained its positive control on **raw** `y`
(`scale/negation_scope.py`, clause 5) while `run_arm` trains every arm on
**standardised** `y` and un-standardises before scoring
(`scale/m3_capability.py:196`, `:189`, `:200`). Both readings were then compared
against the same bar of 1.0. Adam's step is bounded by `lr` almost regardless of
the gradient, so the distance from the initialisation to the label's scale is
spent out of the step budget: at `e2_consequence`'s label sd `0.061984` the
control read `trained_two_feature = 2.446645` at the shipped 150 steps and the
bar printed BROKEN. That reads as *"no arm can pass this task"* when what was
measured is *"the control was handed the label in the wrong units."*
`STATE.md:73-76` diagnosed it and left it, so `e2_consequence` — the one rung
the theory actually predicts on, `t* = 31` against a hop budget of 2 — has
**never been trained** (FINDINGS A7).

**Rule.** The control and the thing it controls must see the same
preprocessing, the same budget and the same optimiser, and the invariance must
be asserted rather than intended. NRMSE is scale-free, so the test is cheap:
scale the label by `c` and require the reading not to move. Fixed and bound by
`tests/cameron/test_bar_control_sees_the_arms_preprocessing.py`; the broken path
is kept reachable as `calibrate_bar(standardise=False)` so the defect stays
measurable rather than merely described.

### M-2. A threshold refitted to the data it judges

`scale/e_ladder.py:23-27` freezes `0.027260` and says why: it is the
pre-registered 13-seed resolution computed from the `negation_scope` pilot SD
**before the data landed**, and is *"deliberately NOT recomputed from the
realised spread, because a threshold refitted to the data it judges is not a
threshold."* The realised spread is printed beside it instead.

**Rule.** Freeze the threshold with its provenance — which pilot, which seed
count, which commit — and print the realised spread next to it rather than
substituting it. This is the one place in the repo where the discipline is
already written down correctly; copy it.

### M-3. Pilot spread taken as the realised spread

The pilot was **2.18× optimistic**: realised `sd 0.109199` against a piloted
`0.050146` (`STATE.md:201`, `DONE.md:339`).

**It is not confined to variance, and it recurs.** The same failure in the cost
domain: `settledrow/settled` measures **`3.48` at `s = 16`** but **`7.43`–
`11.19` at `s = 64`**, against a flat FLOP ratio of `1.700` — so scaling a pilot
by the FLOP model understates the bill by `2×`–`3×`
(`.superpowers/sdd/polymorphic-drifting-squirrel/neptune-report-it2.md:110-118`).
The reason it belongs here rather than in a footnote is that it very nearly
shipped: *"It nearly caught me: the `s=64` projection was drafted from the pilot
ratio before being measured."*

**And the repair is in the code, not in the report that noticed it** — which is
the part worth copying. `scale/m3_flops.py:115-116`, inside the FLOP term
itself: *"A pilot cost measured at small `s` must not be scaled to full geometry
by this term; doing so understates the bill by 2x to 3x."* The next person to
scale a pilot reads it at the point of use rather than having to have read a
round's reports.

In the power domain, the C1 ladder's realised `sd_paired` of `0.019`–`0.109`
against a pilot's `0.050` is what makes `t* = 1` a 62-seed rung (M-9).

**Rule.** A pilot bounds nothing it did not measure. A ratio measured at one
geometry, one `n` or one arm does not transfer to another — report the realised
value against the piloted one every time, treat a ratio above ~1.5 as
invalidating the sample size or the budget the pilot licensed, and put the
warning in the code that does the scaling rather than in the report that
noticed it.

### M-4. A single-seed bootstrap interval read as seed variability

*"Disjoint one-seed bootstrap intervals rule out RESAMPLING noise, not SEED
noise"* (`DONE.md:187`, `STATE.md:211`, which records `sd 0.064106` for
settled).

**Rule.** Name the source of variation the interval covers, in the same
sentence as the interval. A bootstrap over draws at fixed seed answers a
question about draws; seed variability needs seeds, and the count of seeds is
part of the number.

### M-5. A process that cannot cross its own threshold

The old `t=5` e-process unit had a ceiling of `3.80169140625` against a
`THRESHOLD` of `40.0` — *"cannot cross"* (`CHECKLIST.md:1264`, `STATE.md:20`).
The unit was structurally incapable of producing the verdict it was run to
produce, whatever the data said.

**Rule.** **Ceiling arithmetic must be printed before the first number.**
Compute the maximum value the statistic can attain under the design and compare
it to the threshold, as the first line of output. This generalises V-10 from
gates to whole instruments.

### M-6. A partial run read as a verdict

`scale/foreman_looped.py`'s `falsifier()` returns `complete: False` and refuses
a verdict on a partial table; the table holds **2 of 6** cells (FINDINGS D1).
The CUDA lane holds **22 of 60** (FINDINGS D2). `STATE.md:21` still reads "RUN
IN FLIGHT" for the CPU lane, which is complete (FINDINGS D3).

**Rule.** The instrument reports its own completeness and refuses to render a
verdict below it — `foreman_looped` is the pattern. Cell counts go in the
artefact, not in a person's memory of how far the run got.

### M-7. A pre-registration with a hole

`E_LADDER_PREREGISTERED_READING.md` §6 rows A–G all conditioned on `settled`
winning somewhere, so *"a settled arm that only ever loses fell through all of
them."* Row H was added at 13:35, before the numbers, with the timestamp
disclosed (`CHECKLIST.md:1268`).

**Rule.** Rows must be **exhaustive before the data**. Enumerate the outcome
space and check every branch of `verdict()` has a matching row and every row a
matching branch. Adding a row after the data exists is a different act from
adding one before, and the timestamp is the only thing that distinguishes them —
so disclose it.

---

### M-8. Pricing every arm at one arm's rate

Two independently sourced instances, in one round, in two different documents:

* `STATE.md:21` prices a rung at roughly 77 min, which is about **twelve
  `settled` cells** — but a rung is 5 `settled` + 5 `twin` + 5 `softmax`. The
  four-rung ladder measures **`6,047 s = 1.68 h` against its `~5 h`**:
  over-priced **`3.0×`** (`results/r9_pricing.md:175`, and M22 at
  `.superpowers/sdd/polymorphic-drifting-squirrel/mercury-report.md:49`).
* `results/r9_systems_gate.md:196` prices ten units as ten `settled` cells —
  the `29.6 h` of P-8. Re-priced at per-arm rates it is **`15.5 h`**, and the
  headline ratio moves from `34×` to `13.5×`.

Mercury records this as the **third** instance of the type in a single round
(`.superpowers/sdd/polymorphic-drifting-squirrel/mercury-report-it2.md:216`).
The two above are the ones with a document and a number attached here; the third
is the subject of his own §4.4 and is not separately cited in this entry, so
treat "three" as his count and "two" as what this file evidences.

The cells are not interchangeable and the code says so:
`need_gram = self.base_cell == "settled"` (`scale/m3_quintuple.py:404`, `:444`),
so `twin` skips the Gram entirely and has no settle loop.

**And the ratio does not transfer across tasks, or even keep its sign.** Mercury
measured `twin` at **`1.61×` dearer** than `settled` on `negation_scope` at
`ntr8192` (medians `508.74 s` vs `315.65 s`; `1.23×` on minima), and **`2.1×`
cheaper** at `e3_t1`/`ntr2048`. So "charge everything at the dearest arm's rate"
is not even a safe over-estimate — which arm is dearest depends on the task, and
the correction to the `29.6 h` figure runs the *other* way on its baseline for
exactly that reason.

**Rule.** Price each arm at its own measured rate. Never carry a cross-arm cost
ratio across a task, a geometry or an `n`, and never assume the substitution is
conservative — a ratio that inverts makes the "safe" direction wrong. If only
one arm has been timed, the estimate covers that one arm and says so.

### M-9. A verdict whose finest achievable p cannot reach the α it quotes

M-5 in the inference domain, and it is general to every reading in this
repository rather than to one instrument. At **N = 5 seeds the verdict is a sign
test.** Measured on the shipped `contrast()` over 1,000 samples
(`.superpowers/sdd/polymorphic-drifting-squirrel/progress.md:1013-1021`):
unanimity excludes zero **385/385**; a 4–1 split excludes it 20–44 % of the
time; a 3–2 split 0–3.7 %. So *"the CI excludes zero"* at five seeds is very
nearly *"all five seeds agreed"* — and **the finest achievable two-sided p at
N = 5 is `0.0625`, not the `0.05` the project quotes.** The design cannot
produce the significance level it reports, whatever the data say. This covers
**every 5-seed reading here, including the standing `+0.108437` headline**
(`CHECKLIST.md:1168`, `ceq/hf_artifact/README.md:39`).

**Half the repair is already in the tree, which is what makes the rule
concrete.** Both of those rows already print the seed-agreement count beside the
interval — `ceq/hf_artifact/README.md:39` reads
`+0.108437 | [+0.066232, +0.147110] | 5/5`. The practice exists; what is missing
is that it is not required, so a number quoted anywhere else loses the `5/5` and
with it the only signal that the interval is a sign test. A convention followed
in two places and mandated in none is a convention that the next headline will
drop.

**RESOLVED, and it was not a transcription error.** The disagreement noticed
here turned out to be systematic across all three headline contrasts — every
point estimate and every upper bound agreeing, lower bounds differing — and both
families have live producers. At five seeds the paired resample space is FINITE:
`5**5 = 3125` tuples with **126 distinct means**, so the exact percentile is
computable and is a different estimator from a Monte-Carlo draw of the same
distribution. Reproduced from `results/m3_quintuple_v2.jsonl`:

    contrast            B=10000 seed=0          exact over 3125
    settled - twin      [-0.048587, +0.031557]  [-0.042903, +0.031557]
    settled - softmax   [+0.066232, +0.147110]  [+0.068181, +0.147110]
    argmax  - softmax   [-0.134115, -0.102204]  [-0.134115, -0.102786]

The first column is what `ceq/hf_artifact/README.md` ships; the second is what
the root `README.md` and `CHECKLIST.md` print. **Neither number is wrong. No
number is orphaned.** `tests/cameron/test_published_intervals_have_producers.py`
binds both, and rejects the mis-pairing.

**So it is not P-1 — it is P-8, recurring, and that is the sharper finding.**
The generator's own limits paragraph (e) already states the whole thing, in the
file that ships (`ceq/hf_artifact/README.md:85`), naming both families and both
endpoints. The table three lines above it still prints `95% CI` with no
estimator, and the JSON behind that table **already carries `n_boot: 10000` and
`boot_seed: 0`** — the provenance exists and is dropped at render time. A caveat
that is correct, complete, in the right place and below the row that gets quoted
is the exact failure P-8 describes, and this is it happening to the
repository's most-quoted result while P-8 was being written.

**AND THE RENDER-TIME DROP IS NOW CLOSED, which is the half a diagnosis does
not fix.** The card's contrast table names its estimator in the column header
and carries, beside it, the journal records the row was computed from — the
keys read from the journal rather than reconstructed from the arm and the
geometry, which would be a second copy of a grammar that has one parser.
`tests/cameron/test_headline_ci_provenance.py` recomputes both resampling rules
from the journal and pins each published endpoint to the estimator its own
document claims.

**One correction to this entry's own first repair.** That repair rendered the
two endpoints as adjacent columns, on the reading that they were two estimator
families. They are not. The paired resample space at five seeds has at most
`C(2n-1, n) = 126` distinct atoms; the Monte-Carlo draw lands on atom 7, the
exact percentile is atom 8, and `ci_hi` is atom 117 for both. **One estimator
reaching adjacent atoms, not two families** — so printing them side by side as
rivals asserted a distinction that does not exist, and the column was withdrawn
in favour of naming the instrument once and printing the atom count. Recorded
here because the first fix for a labelling failure was itself a labelling
failure, in the opposite direction: under-labelling invites a typo reading,
over-labelling invents a second instrument.

**The general shape, and it is not about bootstraps.** Two correct numbers that
disagree are indistinguishable from one correct and one wrong number unless each
carries the procedure that produced it. An unlabelled number is not merely
undocumented — it is *unfalsifiable*, because there is no claim to check it
against. P-8 says the caveat was present but below the number; this adds that a
caveat one render away from the number is already too far.

The same round produced a second face of it: realised `sd_paired` on the C1
ladder is `0.019`–`0.109` against a pilot's `0.050`, so `t* = 1` needs **62
seeds, not 5** — underpowered by an order of magnitude at one rung while the
others resolve. The response on record is the right one and is worth copying:
run the rung, label it UNDERPOWERED with its seed requirement in the same
breath, and show the whole ladder — because dropping the rung that cannot
resolve and reporting only the rungs that can is rung-picking.

**Rule.** Compute the design's **finest achievable p** and compare it to the α
being claimed **before the run**, exactly as M-5 requires the statistic's
ceiling to be printed before the first number. And print `n+`, the seed-agreement
count, beside every interval — at small N it is what the interval is actually
reporting, and a reader who can see `5/5` cannot mistake a sign test for a
bootstrap.


## D — Design-level failures

### D-1. Racing a baseline at its proven optimum

**This is the largest one in the repository and it subsumes most of the null
results.** Every label here is a scalar point prediction at position `s-1` —
`return out if self.vector_readout else out[:, s - 1]`,
`scale/m3_quintuple.py:483` (FINDINGS C1). **Amended R9: the escape this entry
asks for now EXISTS and is off by default.** `vector_readout`
(`scale/m3_quintuple.py:368`, documented at `scale/m3_quintuple.py:355-359`)
drops the `[:, s-1]` index; it defaults to `False`, so every shipped number
here is still the scalar-readout shape and the argument below is unchanged for
them. The entry previously cited `:311`, which is a BLANK LINE — the quoted
source had moved and the checker could not see it, because a blank line is in
range. That hole is closed in
`tests/cameron/test_mistakes_citations_resolve.py`. One softmax layer is **provably Bayes-optimal on exactly that shape**,
where linear attention provably cannot be (`arXiv:2410.01537`, ICLR 2025;
conceded at `LOOP_PROMPT.md:34-38`) (FINDINGS C2). The repo already knew:
`STATE.md` items 27-28 record *"every task here asks for a point prediction —
which is exactly the single-location problem where one softmax layer is provably
Bayes-optimal"*, and the novelty claim is *"UNTESTED because no vector-valued
label exists here"* (FINDINGS C3). The waste is visible in the code:
`impact_truncation` accumulates `Σ_h (ρA)^h Bn` over the whole graph and then
returns `y[b] = acc[query]` — one coordinate (`impact_truncation`,
`scale/impact.py:754`, FINDINGS C4).

**Rule.** Before running an arm against a baseline, state the regime in which
the baseline is optimal and check the task is not inside it. If it is, the
result is a theorem, and running it buys nothing. Dropping the `[:, s-1]` index
yields an `[n, s]` vector label for **zero new parameters** — the same
`readout` at every position — and leaves that regime. It must go in a separate
lane with its own journal and weights directory, because changing `forward`
voids the `PUBLISHED_SOFTMAX_8192` reproduction gate at
`scale/m3_quintuple.py:696-708`.

### D-2. An oracle that is the arm's own resolvent

The `e3_t*` oracle is the arm's own resolvent, so `settled − softmax` is **VOID**
on that ladder; only `settled − twin` is creditable, and it reads `−0.002959`,
CI `[−0.042903, +0.031557]` (`results/e_ladder_reading.txt`,
`LOOP_PROMPT.md §1.7d`, FINDINGS C5). `E1` is the same shape and the registry
says so at `scale/negation_scope.py:1033-1041`: its label is the signed path sum,
*"the object the ceq resolvent computes, so an arm built on that resolvent is
being asked to reproduce its own forward"* — kept only as a must-fire, crediting
nothing.

**Rule.** Write down what computes the label and what computes the prediction.
If they are the same operator, the comparison is a reproduction check and no
number from it is a capability claim. Say which contrasts are VOID *in the
pre-registration*, not after reading them. E1's registration comment is the
model to copy.

### D-3. A difficulty dial that does not vary

Rank was retired as a difficulty column with a proof: `z' = a·z + b` is a
2-dimensional linear recursion, so ring-automaton rank is **exactly 2 across the
whole E3 ladder while `t*` runs 1 → 63** (`CHECKLIST.md:1292`,
`STATE.md:78-79`). A constant column spanning the entire difficulty range
describes nothing.

**Rule.** A metadata column claimed as a difficulty axis must be computed across
the full range of the ladder and its spread reported before it is used to
order anything. Constant across the range means retired, with the replacement
named — here, `t*`.

### D-4. Registration without admission

`E_T_STAR` (`scale/negation_scope.py:1017-1025`) has no entry for `impact`,
`impact_hetero` or `e4prime`. `scale/e_ladder.py:143` and
`scale/etask_k5e.py:117` index it unguarded and raise `KeyError`;
`scale/m3_capability.py:269` and `scale/m3_quintuple.py:652` guard with `in` and
**degrade silently instead** (FINDINGS A3). A task can therefore be registered,
run, and produce a reading with its difficulty dial simply absent.

**Rule.** Two consumers of one registry must fail the same way. A silent
degrade beside a hard `KeyError` on the same missing key means the reading you
get depends on which caller you used. Where a registration surface has required
side-tables, validate the entry against all of them at import — or accept that
`M3_TASKS` being a plain dict is itself the finding (FINDINGS F).

### D-5. Declaring `0.0` without a movement test

Flipper dependence is a **two-sided band** (tol 0.05) and is the *wrong-task*
check, not the anti-vacuity check. `e4prime` and `impact` both declare exactly
`0.0` (`impact_flipper_dependence`, `scale/impact.py:775`). A task declaring `0.0` must ship a do()-bit
movement test, or the zero is indistinguishable from the vacuous kind that
already got `unshocked_equilibrium` rejected
(`tests/cameron/test_m3_etasks.py:356`) (FINDINGS F).

**Rule.** A declared zero is a claim that nothing moves the label, and it needs a
positive showing what *does*. Ship the movement test in the same commit as the
declaration.

---

### V-14a. The scope test that condemns every refusal guard

Round 10 mechanized V-14's question as a set census: for each control, compute the
reachable set `R(control)` against the production domain `D`, and fire iff
`R n D = empty`. Over 31 controls it fired **19 times and 17 were not defects**.

**A refusal guard is exercised on inputs production cannot emit BECAUSE THAT IS
WHAT A REFUSAL GUARD IS FOR.** Planting a violation to prove a check can fire
requires an input the production path would never produce; that is the plant
working, not a scope failure.

**`R n D = empty` is necessary, not sufficient.** The clause has to be conditioned
on the control's ROLE, filed before scoring:

- **certifying** -- asserts a property OF PRODUCTION. Must enter through the
  production path. An empty intersection here is the defect V-14 describes.
- **excluding** -- proves an instrument REFUSES something. Constructed inputs are
  correct; its job is to reach a branch, not to describe production.
- **illustrating** -- neither certifies nor excludes; scored as neither.

**THE REPAIR CARRIES ITS OWN HOLE, AND IT MUST BE STATED WITH IT.** Role is
*declared, not measurable*. A mislabelled role gets a wrong verdict silently, and
nothing in the census can detect the mislabelling. The taxonomy converts an
over-firing rule into a correct rule that depends on an unverifiable input.

**IT ALSO RESOLVES A CONFLICT BETWEEN TWO RULES IN THIS FILE.** V-16's repair --
and round 10's instance 21 -- says *bind a must-fire to a CONSTRUCTED input* so it
cannot expire when live data changes. V-14 says a control whose reachable set
misses production certifies nothing. **A must-fire bound to a constructed input is,
by construction, a control whose `R` may miss `D`.** Both rules are right and they
apply to different roles: the first to *excluding* controls, the second to
*certifying* ones. Without the taxonomy they contradict, and a reader obeying both
is stuck.

### V-15. The condemning rule with no planted negative

Rule 5 requires a planted **positive** for a reported absence: an instance that
carries the property, to prove the instrument can find it. That check is
powerless against a rule that CONDEMNS, because a condemning rule's failure mode
is firing on the innocent, and every planted positive it fires on is a success.

Round 10 catalogued nineteen instances of one mechanism — a rule keying on a
proxy instead of the property it names (`R10_MECHANISM.md`). **Seven of the first
fourteen condemn rather than excuse**, and a planted positive cannot catch any of
them: they fired enthusiastically, on the wrong thing.

The missing check is the mirror of rule 5. **A rule that condemns needs a planted
NEGATIVE: an instance that carries the proxy and is innocent of the property.**
If the rule fires on it, the rule is keyed on the proxy.

Worked example from the same round. `axis_rho` was scored by comparing two
argmins, and the guard written against it would have fired a RED on `depth` — a
trend axis whose argmin sits at a grid edge *by construction*, and therefore
innocent. The allowlist that prevented it was added before `depth` ran, on
reasoning rather than on a planted negative. Had the order been reversed, a
correct measurement would have been condemned by a guard written to catch
condemnation.

### V-16. The instrument that cannot measure, reporting a pass

An instrument has three outcomes, not two: the property holds, the property
fails, and **it could not tell**. Collapsing the third into either of the first
two is a defect, and collapsing it into "pass" is the one that ships.

    return int(out) if out.isdigit() else 0     # unreadable -> "nothing is running"

Measured, round 10 iteration 11. A process-count probe mapped every unreadable
result — empty output under memory pressure, a timeout, a transient failure — to
`0`, and `0` licensed the next job to allocate. An absence of evidence was
returned as evidence of absence, in the field a scheduler used to decide whether
to spend 7.9 GiB. It fired: a second seat launched while the first was live.

**A retry loop does not fix this.** Three consecutive unreadable answers are three
consecutive zeros; adding attempts to a fail-open predicate keeps it fail-open.
The two repairs are independent — unknown must be a distinct value that counts as
*not passing*, and where the reading must persist, it must persist as a *measured*
value.

The check is a **planted unreadable**: feed the instrument an input it cannot
measure and confirm it says so. `scale/vram_gate.py` gets this right by
construction, returning UNKNOWN rather than GREEN when `nvidia-smi` or `psutil`
is missing — and the fail-open version above was written in the checker that
decides whether to *call* that gate. Knowing the rule and writing it down is not
the same as applying it one layer up.

### V-17. The threshold imported out of its units

"Import a threshold that already exists rather than picking one that flatters your
result" is a good rule and this repo enforces it -- `scale/r10_dual_oracle.py`
imports `kirchhoff.AGREEMENT_TOL` by **object identity**, so a local re-pick fails
the self-check. The rule has an unstated precondition, and round 10 paid for it.

Two imports, both principled, one sound:

**Sound.** `rips_gate.FAIL_BAR = 0.9`, defined as "the decoder is doing
essentially nothing" and already used in that role at `impact.py:1121`, imported
for a within-split probe clause. Same quantity, same role, same units.

**Not sound.** `delta = 0.5`, taken from the chain task's
`flipper_dependence > 0.5` clause and applied to a harmonic corpus. Measured:
`fd_max x |B|` is near-constant at **3.741** across twelve rungs, because one
boundary node's influence is `~1/|B|`. So `fd >= 0.5` requires `|B| <= 7.5` while
the corpus runs `|B| = 4..80`. **Ten of twelve rungs fail by construction.** The
source threshold is worse than inapplicable: the chain's own closed form
`2/sqrt(t*)` drops below 0.5 at `t* > 16`, so that clause fails on the chain
itself at `t*=32`.

**THE TEST, and it is one question.** Ask what value the quantity takes under a
null or trivial predictor.

- **Anchored** -- NRMSE is RMSE over `std(y)`, so the mean predictor reads exactly
  1.0 for any task at any scale. A threshold in NRMSE means the same thing
  everywhere and travels.
- **Not anchored** -- `flipper_dependence` is `|label move| / |label scale|`, whose
  achievable range is set by the generator's parameters. A threshold in those units
  is meaningless away from the construction that produced it.

An imported constant carries the authority of having been used before, which is
the credibility-transfer defect one field over: the number is real, the source is
real, and neither fact makes it applicable here.

### V-18. The guard that only guards its callers

`scale/it11_verdict.py` exists because the it.8 journals carry bit-identical
duplicate rows, so a reading that counts rows reports `N=2` where one seed was
run. The module's `by_seed` deduplicates, `verdict()` refuses below `N=8`, and its
docstring names the defect as instance 16. That repair held for every caller.

**It did not hold for a reader that was not a caller.** An ad-hoc analysis
written in the same round, by the same author, one iteration after the module
shipped, regrouped the journal inline with a dictionary keyed on `n_train` and
read the mean over ROWS. It reported `N=9` at n=2048 and `N=10` at n=32768 where
eight distinct seeds exist, and put the interpolated crossing at **12,789**
against the seed-correct **12,780**.

The arithmetic damage was 9 in 12,780, about 0.07%, and that is the point rather
than the excuse. **The magnitude of the error is set by how duplicated the journal
happens to be, not by anything the reader controls.** The same bypass over a
journal with three duplicates of one seed would have moved the answer by a
multiple of that, silently, and the reading would have looked identical.

**THE TEST.** For every reading that consumes a journal a shared reader already
covers, ask whether it *calls* that reader or re-implements it. Grep for the
grouping key -- here `seed` -- across analysis scripts and confirm each hit routes
through the module rather than around it.

**Why the usual repairs miss this.** A must-fire proves the guard fires when
called. A planted negative proves it fires on the right thing. Neither says
anything about a code path that never reaches the guard, and the round's own scope
clause (`R n D`) scores controls, not readers. An instrument is not a policy: it
constrains only the paths routed through it, and every inline re-implementation is
a path that is not.

The shape generalises past this repo. A repaired instrument reads as a repaired
repository, which is instance 23 in `R10_MECHANISM.md`; this is that instance in
its narrowest form, where the bypass is a single dictionary comprehension and the
author is the person who wrote the guard.

### V-19. A wait-gate polling for work that had already finished

`/tmp/waveB.sh:5` computed its drain condition as
`n_wrote=$(grep -hc 'WROTE' results/r10_it8_waveAC_t*.log | paste -sd+ | bc)`.
`bc` is not installed on this host: `which bc` returns nothing across all forty
PATH entries. The substitution yielded the empty string, and line 7's
`[ "$n_wrote" -ge 6 ]` raised `integer expression expected` on every pass rather
than evaluating false. A test that errors is not a test that fails; the loop
carried no branch for the error, so it slept 45 seconds and retried without
bound, from 20:40:20 on 2026-08-30 until it was killed at 01:02 the next
morning.

What it was waiting to start had already run. `results/r10_it8_waveB.log` holds
the completed wave at lines 16-26 — `t*=2 n=32768 eval NRMSE=0.874834 [LEARNS]`
`boot[0.8620,0.8903] 624.5s`, `t*=8 ... 0.972372 [LEARNS] boot[0.9641,0.9814]`
`607.8s`, `t*=32 ... 1.006066 [NO READING] boot[1.0033,1.0094] 680.0s`, then
`WAVE B COMPLETE` — and the first `bc: command not found` appears at line 30,
after it. The journals corroborate: `results/r10_it8_capacity_softmax_t8.jsonl`
and `..._t32.jsonl` each carry `n_train=32768` cells at seeds 0 through 7, a
complete N=8. The gate spent four hours and twenty-two minutes polling for
permission to begin work that was already on disk, and appended 43,974 bytes of
its own error text to the very log that recorded the completion.

The mechanism is therefore not lost data. It is a gate whose predicate could
neither pass nor fail, attached to a job with no check for whether it was
already done.

**Check:** a wait-gate must distinguish *condition false* from *condition
unevaluable* and exit loudly on the second; compute the gate quantity with a
tool the host is known to carry, since `awk` and shell arithmetic exist wherever
`sh` does and `bc` does not. Separately, a job that appends to a log must read
that log's head before starting, not its tail.

### V-21. A tail read as if it were the file

The first record of V-19 in this document asserted that wave B "never ran" and
that `n=32768` was "the point that never ran", naming it as the cost of the
defect. Both statements were false when written. They came from reading the
last 600 bytes of `results/r10_it8_waveB.log`, finding `bc: command not found`
repeated, and generalising the tail to the file. The completed wave sat at lines
16-26 of the same file, unread.

The error survived one round of apparent corroboration. A process table showed
the broken script alive, a `which bc` returned nothing, and the gate arithmetic
was confirmed unevaluable — three checks that all agreed, and none of which
touched the question of whether the work had already happened. Agreement among
checks that share a blind spot is one check.

A second-order defect followed from the first: acting on the false conclusion
launched a duplicate sweep that recomputed `t*=2 n=32768 seed=0` to
`eval_nrmse 0.8748335142818684`, bit-identical to the row already journalled,
and appended header, ceiling and bar rows to three journals before it was
killed. Those rows were reverted at `git checkout -- results/`.

**Check:** before asserting what a file does not contain, grep the whole file
for what would contradict the assertion. `tail` answers what happened last, and
a claim about whether something ever happened is not that question. When several
checks agree, name the question each one could not have answered.

### V-20. A cap requested in prose, and a sentinel that reads it as infinity

A loop was mounted with a request for a hard ceiling of 60 iterations and
mounted unbounded. The request never reached the parser: the invocation carried
the cap four times as prose — `max_iterations 60`, `hard cap`, `terminate at
60`, `max_iterations MUST be 60` — and never once as `--max-iterations 60`.
`setup-ralph-loop.sh:25` recognises only the literal flag form, so the prose
fell through the catch-all argument branch at `setup-ralph-loop.sh:141-145`
into the prompt body, and `MAX_ITERATIONS` kept its initialiser `0`
(`setup-ralph-loop.sh:10`). The banner then printed `Max iterations: unlimited`
and `This loop cannot be stopped manually`, both accurate.

The second defect is what made the first one silent. `stop-hook.sh:61` guards
the ceiling as `[[ $MAX_ITERATIONS -gt 0 ]] && [[ $ITERATION -ge
$MAX_ITERATIONS ]]`, so `0` means unlimited — and `0` is also the value left
behind when a cap is requested in a form the parser does not recognise. "No cap
was asked for" and "a cap was asked for and lost" are the same byte, and the
one the machine prefers is infinity. A third form, `--max-iterations=60`, drops
silently the same way.

A related field is decorative rather than wrong: `stop-hook.sh:13-18` tests only
whether the state file exists and never parses `active:`, which is why the
earlier loop deactivated at `ecbedf7` was in fact stopped by renaming the file,
and why the `active: false` that commit credits had no effect.

Both were repaired and the repair verified by execution — 8 mount cases, 6
stop-hook cases, an end-to-end run where a cap of 3 terminates on the third
stop. `--max-iterations` is now mandatory and must be a positive integer or the
literal `unlimited`; prose, `abc`, `sixty` and bare `0` each exit 1 writing no
state file; `active: false` now halts. The patch lives in
`plugins/cache/claude-plugins-official/ralph-loop/1.0.0/` and a plugin update
reverts it, along with an earlier hand patch already present in that cache and
absent from the marketplace tree. Full account in `V13_RALPH_LOOP_FIX.md`.

**Check:** after any mount that accepts a bound, read the bound back out of the
state the machine consults, never out of the request — the request can be
well-formed prose and still be invisible. Where a sentinel doubles as a parse
failure, that read-back is the only thing separating them.

### P-9. A commit message that contradicts the commit before it

`ecbedf7` (2026-08-31 00:40:40) states that the range "stopped at it.20 with
the it.21 n=8,192 wave killed before its first cell landed."

`67162a6` (00:40:12), twenty-eight seconds earlier and by the same author,
committed `results/r10_r10_it21_t8_n8192_capacity_softmax_t8.jsonl`, whose
cell row reads `"t_star": 8, "n_train": 8192, "seed": 0, "steps": 150,`
`"train_nrmse": 0.9732971180581712, "eval_nrmse": 1.0252097125261737,`
`"boot_lo": 1.0168885291722383, "boot_hi": 1.0339123632622422,`
`"verdict": "NO READING", "secs": 97.3`.

The first cell landed, ran 97.3 seconds, and returned NO READING. What was
killed was the remainder of the wave: the same journal's final row records
`"t": "dropped", "steps": [600, 2400, 9600], "why": "unaffordable, see priced`
`DAG"`. The message conflated *the wave was truncated after its first cell*
with *the first cell never landed*, and the distinction is the whole of the
result — a cell that returns NO READING is a measurement, and a cell that
never ran is an absence.

**Check:** a commit message that describes what did not happen must name the
artefact it checked to establish the absence. Where the artefact is a journal,
quote the row, or quote the empty result of the grep that found no row.

### D-6. The repair that was written and never started

`/tmp/waveB2.sh`, modified at 20:50 on 2026-08-30, is `/tmp/waveB.sh` with the
V-19 defect removed: it counts with `grep -h 'WROTE' ... | wc -l | tr -d ' '`
and guards the comparison as `[ "${n:-0}" -ge 6 ]`. Both changes are correct
and either alone would have released the gate.

It was never run. The broken script, started at 20:40:20, was still the only
live process at 01:02 the following morning, four hours and twelve minutes
after its replacement was authored ten minutes downstream of it. No process
for `waveB2.sh` appears in the process table at any point, and
`results/` contains no log it would have written.

The failure is not the bug; the bug was found and fixed inside ten minutes.
The failure is that fixing it and deploying it were treated as the same act,
so the diagnosis was recorded on disk while the diagnosed process kept
running.

**Check:** a repair to a running process is not applied until the old process
is dead and the new one has emitted its first line. Kill first, then start,
then read one line of output before moving on.

### M-10. A finding filed without grepping for the guard that already existed

The first version of this entry reported the thread-count effect on
`eval_nrmse` as a new discovery and asserted that "nothing in the protocol pins
the value". Both halves were wrong, and the evidence was one grep away.

The effect is real and was reproduced here — `t*=2, n=2048, seed=0, steps=150`
returns `0.971432426855954` at `threads=8` and `0.9690874072329486` at
`threads=6`, both bit-for-bit against the journal — but the repository found it
first and had already built the guard. `scale/it11_verdict.py:129-165`'s
`by_seed` states the mechanism at equation level in its own docstring, names the
drift as `2.345e-3` across all three cross-thread pairs
(`2.345e-3, 4.911e-4, 7.311e-6`), builds every seed interval at a fixed thread
count, picks the count carrying the most distinct seeds, and raises on a
same-seed same-threads disagreement. `thread_split` reports what was dropped.
`results/r10_inspector_phase1a.md:25` verified the floor independently and
marked it CLEAN; `R10_MECHANISM.md:370` already catalogues *misreporting* that
floor as its own defect; `R10_ITERATION_10.md:142` warns against comparing a
7e-5 quantity against it "as if they were the same kind of uncertainty".

What survives is narrow and prospective. The equivalence margin
`Δ_eq = 0.5 σ_softmax-seed` that v-main.6 introduces appears nowhere in the
tree, so the thread floor has never been compared against *it*: at
`t*=2, n=2048` the N=8 seed sd is `0.010101`, giving `Δ_eq = 0.005051`, and the
`0.002345` floor is `0.464` of that. A margin under roughly `4.7e-3` is smaller
than twice the reduction-order floor and cannot be defended. Separately,
`r10_capacity_sweep.py` defaults `--threads` to 8 while the recorded N=8 wave
ran at 6, so a cell added with default flags will not join the existing spread —
`by_seed` will silently drop it into the minority thread group rather than
corrupt the interval, which is the guard working, but the cell is then wasted.

**Check:** before filing a defect, grep for its guard. The instruction to do so
was already standing in this round's own loop prompt — *before asserting a file
lacks something, grep the whole file* — and this entry is what skipping it
produces: a rediscovery written in the voice of a discovery, which costs more
than silence because it invites re-fixing something already fixed.

### M-11. Whitening checked at lag 1, integrated at frequency zero

A sequential detector was calibrated on an independent null and deployed on a
dependent one. The closed-form threshold agreed with simulation to 0.71% where
the assumption held, which is what made the failure invisible: at the measured
first-order autocorrelation `φ̂ = 0.709` the realized average run length to
false alarm was `41.5` against a nominal `1000`, a `24.1×` shortfall, and even
a mild `φ = 0.3` gave `149.1`. End-to-end at the deployable operating point the
nominal `22000` was realized as `1459`, a 53.0% chance of at least one false
alarm per run.

The standard remedy was applied and the standard diagnostic passed while the
defect survived. After AR(1) pre-whitening the residual's lag-1 autocorrelation
read `−0.0999`, which any whiteness check accepts, but its Bartlett long-run
variance ratio climbed to `2.78` at bandwidth 200. A cumulative statistic
integrates the spectral density at frequency zero, not the autocorrelation at
lag 1, so a residual can be white by the usual test and still carry the power
that matters. The sliding window that produced the residual writes correlation
at a range of order `W + G`, longer than the burn-in available to measure it.

The same geometry silently filters signal as well as noise. A trailing-window
detrender with a guard-to-transition-width ratio of 0.5 passes only 13% of the
transition amplitude; an earlier revision lost 5 of 8 planted seeds to exactly
that, with no diagnostic naming the cause. At `W=300, G=150` it passes 76.8%.

**Check:** for any statistic that accumulates, validate the null by simulating
from the dependence structure actually present, not by testing residuals for
whiteness at lag 1. Report the long-run variance ratio at a bandwidth of the
window's own scale. And before trusting a detrender, measure what fraction of a
planted signal survives it.

### V-22. A pre-registered constant carried in from another system

A contract pre-registered the worked figure `α = 0.2`, with an uncertain
fraction of `0.398` at `ε = 1e-2` and `0.251` at `1e-3`, as the number the
planted fractal bed would be built to read. Two defects sit inside it.

The exponent is self-consistent and the levels are not measured. `0.398 / 0.251
= 1.58566` against `10^{0.2} = 1.58489`, so the per-decade *ratio* does follow
from `α = 0.2` alone. The levels do not: `(10^{-2})^{0.2} = 0.398107` and
`(10^{-3})^{0.2} = 0.251189` to six figures, which is the scaling law evaluated
with its prefactor silently set to 1. The published law is `f(ε) ~ C·(ε/L)^α`
with a system-dependent `C`, printed explicitly as `n_k/ñ` by Daza et al.
Setting `C = 1` asserts that every state is uncertain at unit resolution, which
is a claim about the bed, not a consequence of the scaling law.

`α = 0.2` is itself the forced damped pendulum's value, from `D₀ ≅ 1.8` with
`D = 2` in Ott's Scholarpedia article on basins of attraction. Pre-registering
it imports one system's boundary dimension into a prediction about a different
system that had not been built yet. A pre-registration is supposed to bind the
analyst to a number derived from the design under test; a number carried in
from an unrelated system binds nothing and will be met or missed for reasons
that have nothing to do with the hypothesis.

This is V-17's mechanism — a threshold imported out of its units — moved one
level up, from a decision threshold to a predicted value.

**Check:** every pre-registered constant states which system produced it and by
what derivation. If the answer is another system's published figure, it is a
prior, not a prediction, and must be labelled as one. Where a law carries a
prefactor, either measure the prefactor or register only the quantity the law
determines without it — here, the ratio, never the level.

### M-12. A calibrated model whose untested branch went unchecked

`ceq/sizing.py` opens by insisting that "a sizing model that has never been
compared against a scale is a spreadsheet", and its live path earns that. Run
against the CUDA allocator on the card it was calibrated on, at `d=256,
heads=4, seq=512, bs=8, fp32`, its non-checkpointed predictions land at
`1.11 / 1.01 / 0.99` of measured for softmax at `L = 4 / 8 / 16` and
`0.98 / 0.94 / 0.91` for the signed arm.

Its checkpointed branch was never measured, and it is wrong. The branch returns
`resid / L + C_OPERATOR * op_one + head`, which is independent of `L` once the
residual term is divided out, so it predicts `204.8 MiB` for the signed arm at
every depth. Measured peaks grow with depth: `289.0` at `L=4`, `329.1` at
`L=8`, `409.2` at `L=16`, so the ratio of measured to predicted climbs
`1.41 → 1.61 → 2.00`. The same branch is applied unchanged to the softmax arm,
where it returns the non-checkpointed figure and therefore over-predicts by up
to `4.8×` (`243.7` measured against `1160.0` predicted at `L=16`).

The failure is not the coefficient but the coverage. The docstring's measured
table has five sequence lengths and two arms and no checkpointed column, so the
one branch with no row in the table is the one that drifted. A module can be
scrupulously calibrated and still ship an unmeasured path, and that path will
be the one a memory-constrained plan reaches for first.

**Check:** every branch of a model that returns a number needs a row in the
calibration table, and a branch whose prediction is constant in a parameter the
measurement varies is a claim to test, not a simplification to accept. Where a
model has an `if`, the test needs both sides.

*Recurrence, one entry later.* `Cap.__str__` in `scale/it11_verdict.py` formatted
`h_hat` unconditionally and raised `TypeError` on every cell above the bar, where
`h_hat` is deliberately `None`. Its demo asserted `r3.h_hat is None` but never
rendered `r3`, so the branch had an assertion and no row — the same shape as the
sizing model's unmeasured checkpointed path, committed while that entry was being
written. Asserting a value and exercising the code that consumes it are different
tests, and the second is the one that catches formatting. The demo now prints
every case it asserts.

### M-13. An equivalence margin registered without a reachability check

v-main.6 fixed the parity grammar as TOST with a pre-registered margin
`Δ_eq = 0.5 σ_softmax-seed`, at `α = 0.05`, on the round's standing `N = 8`. The
three constants were chosen separately and their joint consequence was never
computed.

The 90% confidence interval for a two-sample contrast has half-width
`t₍.₉₅, 2N−2₎ · σ · √(2/N)`. Dividing through by `σ` makes the comparison pure
arithmetic against the margin's `0.5`: at `N = 8` the half-width is `0.8807`, at
`N = 16` it is `0.6001`, and it first drops below `0.5` at `N = 23`. So for every
`N` the round actually runs, the interval is wider than the window it has to sit
inside, and **two bit-identical arms return NO VERDICT**. The test cannot emit
its own passing verdict.

Adequacy is a further step beyond reachability. TOST power at a true difference
of zero — the probability of correctly certifying genuinely equivalent arms —
reads `0.000` at `N = 8`, `0.0834` at `N = 24`, `0.431` at `N = 40`, and first
clears `0.80` at `N = 70`. An earlier draft printed `0.042` at `N = 24`, which is
the σ-known normal approximation rather than the exact value; the exact figure,
Monte Carlo at 400k replications, and the module's own `tost()` agree at
`0.0834 / 0.0833 / 0.0757`. `N = 70` was checked by three routes sharing no code
and is unchanged. That is `8.8×` the registered seed count. Priced
against the measured cost curve, THE READING's three points at N=8 cost about
`8.9 h` per arm; at `N = 70` the same three points cost roughly `78 h` per arm,
or about ten days for three arms on this host.

This is M-9's mechanism — a verdict whose finest achievable statistic cannot
reach the level it quotes — moved from a difference test to an equivalence test.
M-9 was filed in this same round. The check it prescribes, *compute the
control's expected value before it runs*, was not applied to the margin because
the margin arrived as a definition rather than as a control.

**Check:** a pre-registered margin is a claim about achievable resolution and
must be divided by the design's own standard error before it is registered. Run
the test on two identical inputs first: a design that cannot certify equivalence
between a sample and itself has no passing branch, and every result it later
reports is the failure branch wearing different numbers.

### M-14. A gate whose threshold sits on the edge of its own null

`scale/r10_capacity_sweep.py:94` guards every cell with
`ok = (not bad(r0t)) and (not bad(r0e)) and r0t >= 1.0 and r0e >= 1.0` — the
untrained arm must read at or above NRMSE 1.0 on both splits, or the run aborts
with INSTRUMENT BROKEN and credits nothing. The intent is right: an
initialisation that already beats predict-the-mean makes every later
"improvement" a measurement against a moving start.

The threshold has no tolerance band, and it is placed exactly where the null
distribution ends. Measured over 16 seeds at `t*=2, n_train=2048`, untrained,
with no training performed:

| arm | min | max | mean | seeds below 1.0 |
|---|---|---|---|---|
| softmax | 1.00055844 | 1.03349997 | 1.01178392 | 0/16 |
| pivot_unsigned | 1.00055861 | 1.03350431 | 1.01178596 | 0/16 |
| windowed_signed | 0.99997039 | 1.03467607 | 1.01216808 | 1/16 |

Softmax's own lower edge clears the gate by `5.6e-4`. The arms sit against the
threshold, not above it. `windowed_signed` has a marginally wider spread — measured as full range,
`3.4706e-2` against `3.2942e-2`; the figures `3.468e-2` and `3.350e-2` printed in
an earlier draft are `max − 1.0`, not ranges and its lower tail crosses: seed 2 reads
`0.9999703932724174`, short of the bar by `2.96e-5`, and an eight-seed run
aborted at its third seed.

A constant predictor can never read below 1.0 — offsetting toward any constant
only raises the residual — so a sub-1.0 reading does require some correlation
between the untrained output and the label. At `2.96e-5` that correlation is a
tail excursion of the random initialisation, not a defect the gate was built to
catch, and the gate cannot tell the two apart because it compares against a
bare constant rather than against the null's own spread.

**Check:** a threshold placed at the exact boundary of a null distribution fires
on the null. Measure the null first — here, sixteen untrained seeds costing no
training at all — and set the bound at a stated distance from its edge, so that
the gate's false-abort rate is a number rather than a surprise. Where the intent
is "not meaningfully better than the mean", the threshold must carry the word
*meaningfully* as a quantity.

### M-15. A descriptive statistic reported without its null, and an invariance read as a result

An operator decomposition was measured and reported as the mechanism behind a
performance finding: the second hop `a @ a` carries `0.7550` of its energy in a
rank-1 common mode, against `0.4899` for the partially-routed form, and the
routing was therefore acting as an accidental common-mode filter. The
decomposition was correct. The number was the null.

Computed against random Gaussian logits at the same measured logit standard
deviation (`0.0375`), causal-masked and softmaxed, over 20 seeds:
`cm_full(a@a)` has a null of `0.7550 ± 0.0004` against a measured `0.7549`,
`z = −0.30`. At the readout row the null is `0.8328 ± 0.0006` against `0.8343`,
`z = +2.38`. The headline figure is the baseline to four decimal places, because
`frac(a@a)` is a monotone function of attention sharpness alone and untrained
attention is nearly uniform, which maximises it by construction.

**The invariance was the clue and was recorded as the result.** Two iterations
earlier the same quantity was measured across three tasks, found identical to
four figures — `0.4006 / 0.4006 / 0.4003` and `0.7550 / 0.7549 / 0.7550` — and
that stability was written up as a finding: "a property of causal softmax at
`s=64`, not of what the data encodes." Every word of that is true, and it is the
signature of a null. A quantity that does not move across any condition under
test, and equals what a random operator produces, is the baseline; reporting its
constancy as structure inverts what it means.

**The compounding error is the object it was measured on.** Every one of these
figures came from an **untrained** operator and was used to explain performance
differences between **trained** arms. Attention sharpness changes under
training and the statistic is monotone in sharpness, so the measured quantity
does not describe the operator whose readings were being explained. The trained
readings themselves are unaffected, but the mechanism offered for them was taken
from a different object.

**The survival test was run and the statistic does not survive.** Training one
arm for 150 steps at `t*=8, n=2048`, measuring before and after:

| | logit sd | `cm_full(a@a)` | `cm_row(a@a)` |
|---|---|---|---|
| untrained | 0.0367 | 0.7551 | 0.8328 |
| trained, 150 steps | **2.7645** | **0.4245** | **0.3736** |

Training sharpens the attention logits by `75.4×` and more than halves the
common-mode fraction, at the readout row from `0.8328` to `0.3736`. So the
operator whose decomposition was reported is not the operator whose readings were
being explained, and the gap is not marginal — the reported figure is more than
twice the trained one.

**Check:** a descriptive statistic offered as a mechanism needs a null before it
is written down, and the null must be matched on whatever the statistic is
monotone in — here, attention sharpness, which costs twenty random draws. Where
a quantity is invariant across every condition varied, treat that as evidence it
is a constant of the construction and test it against a random instance before
reporting it. And a statistic measured at initialisation may not be used to
explain a difference between trained models without first showing it survives
training — a check that costs one 150-step run and, here, would have stopped the
entire line of argument at its first measurement.

### M-16. The thread lane broken by the person who filed the rule against breaking it

The `t*=8, n=32768` deciding cell was taken at `threads=12` for seeds 0 through
5. A session crash killed the lane at 6 of 8 seeds, and the relaunch that
completed it used `--threads 14` - chosen because the machine had been freed and
14 was faster, with no thought given to the lane.

Both cells landed and both read well: seed 6 at `0.970655` and seed 7 at
`0.973938`, LEARNS, `597.3s` and `451.1s`. The journal holds eight cells for the
configuration. But `scripts/v13_adjudicate_hop2.py` filters on a fixed thread
count, inheriting the refusal in `scale/it11_verdict.py:129-165`, and reported
`6/8 - INSUFFICIENT`. It was right to. M-10 records that this harness is
deterministic given a thread count and returns a different number across counts,
with a measured drift of `2.345e-3`; pooling `threads=12` and `threads=14` cells
would put a non-seed variance source inside an interval claiming to measure seed
variance.

Two things make this worth an entry rather than a note.

The rule was filed in this same session, by the same agent, three iterations
before the violation. M-10 was written, the lane discipline was recorded in the
work list under "THE READING must therefore run every cell in one lane", and the
`windowed_signed` run was deliberately launched at `threads=6` to match the cells
it would be compared against. Then a crash made speed feel urgent and the rule
did not survive contact with a freed machine.

And the guard held anyway. The refusal is in the reader, not in the writer, so an
agent that forgets the discipline at write time still cannot cash the result at
read time. That is the property worth building for: a rule enforced only where it
is remembered protects nothing, and this one was enforced where it was not.

**Check:** thread count is part of a cell's identity, so a relaunch after any
interruption must read the lane out of the existing journal rather than off the
machine's current capacity. Where a discipline can be violated by a flag, the
guard belongs in the consumer of the data, not in the producer.

### P-10. A source's intro was cited as its theorem

`CEQ_V15_CONTRACT.md:51-55` names this class L-EQ and files it under the
author's own name: the tag `[V]` — page fetched, intro or description matched
— was treated as license for a load-bearing claim, when the claim required
the source's theorem read WITH its hypotheses. The contract asserts *"two-
thirds of the pre-v13 section-5 strikes were `[V]`-as-theorem."*

**The measured fraction is not two-thirds.** `attic/workdonenew.pre-v13.md`
section 5 ("CLAIMS DISPROVED, WITHDRAWN, OR STRUCK", `:160-211`) holds 44
rows; 4 of them record a claim that held (`:190` 4/4 HELD, `:204` WORKS,
`:205` CONFIRMED, `:179` ALREADY GUARDED), leaving 40 genuine strikes. Of
those 40, four are `[V]`-as-theorem — a real source named, its own equation
or hypotheses not honoured:

- **`:177`** "ARL₀ from the closed form is the false-alarm rate" — Siegmund's
  closed form holds only under i.i.d. nulls; at the measured autocorrelation
  `φ̂=0.709` the real ARL₀ misses the nominal by **24.1×**.
- **`:196`** "X₂₇a's box-counting `d` feeding `κ=λ(1−d)`" — every fetched
  statement of the Kantz–Grassberger relation uses the **information**
  dimension `D₁`; the claim substituted box-counting `D₀`.
- **`:200`** X32 "`ρ_P=√2` for a planted antisymmetric A" — `√2` is the
  random-**matrix** ensemble average; an **exactly** antisymmetric `J` gives
  `ρ_P=2`, measured `2.000000` at four sizes.
- **`:201`** X32 "Poincaré–Hopf gives Σ index=1 for the replicator" — the
  theorem needs the field transverse to the boundary; at `μ=0` it is tangent
  everywhere (measured `dz₀ = −0.000000e+00`), so the hypothesis fails.

**4/40 = 10%, not 66.7%.** Rows that look like candidates and are not: X27b
(no source was cited at all — "no source fixes `c`" — fabrication, not
misreading a citation), X27c (the corrected claim leaned on no named source),
X27d (its own row says "Citation owed `[U]`" — nothing was cited to
misapply), X32's Fisher relation (row's own verdict: "CORRECT THEOREM, wrong
quantity" — the citation is not at fault), X28c (the cited relation is read
correctly; the fault is that agreement is then tautological, not that the
source failed to back the claim). Full row-by-row reasoning in
`V15_N2_SATURN.md`, Task B.

**Check.** Every citation entering a load-bearing statement carries `[V-eq]`
— the theorem's statement WITH its hypotheses, plus one numeric instance run
— or it is `[V]` and inadmissible (`CEQ_V15_CONTRACT.md:51-53`, L-EQ).
Greppable: `grep -c '\[V-eq\]'` against `grep -c '\[V\]'` on any document
about to enter a contract; a nonzero `[V]` count on a load-bearing line is a
blocked commit, not a style note. Before trusting a fetched source, name
which hypothesis in its own statement your instance satisfies — "the page
exists and the intro matches" answers a different question than "the
equation applies here."

### M-17. A census that classified the correction record as the defect

The v15 contract's it.1-4 line requires "the sizing model replaced by the
calibrated one everywhere it was cited". A read-only census was dispatched first
to enumerate the sites, since nothing can be replaced before it is found. The
census returned **5 STALE sites across 4 files** and recommended correcting them
in priority order (`V15_N2B_SIZING_CITATIONS.md:25-33`, `:139`).

**None of the five is a use of the uncalibrated model. All five are the record
that the uncalibrated model was wrong.**

| site | what the line actually is |
|---|---|
| `attic/workdonenew.pre-v13.md:185` | the strike row itself — *"6.63× LOW"* — naming `ceq/sizing.py`'s calibrated constants as the fix |
| `workdonenew.md:275` | the same strike row, carried forward |
| `CEQ_V15_CONTRACT.md:32` | *"The sizing line was 6.63x low."* — a past-tense historical fact in a WHERE WE ARE census |
| `V13_CLAIM_AUDIT.md:59` | an audit row whose verdict column reads **CONFIRMED**, evidence `ceq/sizing.py:48,68` |
| `V13_CLAIM_AUDIT.md:60` | the worked instance, `0.0671` vs `0.4449 GB/layer`, verdict **CONFIRMED** |

The one file that could have held a stale *use* — `ceq/sizing.py` — was already
calibrated: `C_OPERATOR = 3.9` at `:48`, `DTYPE_MODES['bf16_autocast'] = (2.2,
3.4)` at `:68`. The census reports this correctly under its CALIBRATED heading.
The defect is confined to the STALE column, where a keyword match on `6.63`
became a verdict.

**Why this is a mechanism and not a typo.** The census searched for the literal
number and classified by presence, so the sentence *"X was 6.63× low"* and a
sentence *using* a 6.63×-low figure are indistinguishable to it. Every
correction record in a repository states the wrong number in order to say it was
wrong. A citation census keyed on the value therefore lands hardest on exactly
the documents that already fixed the problem — the audit trail scores as maximal
defect density.

**What it would have cost.** Had the recommendation been applied, the five
sites carrying `6.63` would have been rewritten to the calibrated figures, and
`6.63` — the discrepancy factor, whose derivation `3.9 × (3.4 / 2.0)` lives only
in those rows — would have had no producer anywhere in the tree. Repairing a
provenance record by deleting it manufactures a fresh **P-1 (a number with no
live producer)** out of a completed fix, and does so invisibly, because the
diff looks like a correction.

**Check.** A census that classifies must distinguish *use* from *mention*. Before
any site is edited, read the surrounding clause and answer one question: does
this line ASSERT the number as current, or REPORT that it was wrong? Mechanically:
a hit whose line also contains a strike verdict (`STRUCK`, `LOW`, `CONFIRMED`,
`was`, `×`-as-discrepancy) or that sits in a table with a verdict column is a
mention until proven otherwise. More generally — **a search keyed on a value
cannot classify that value's role, and any census reporting a STALE count
without a use/mention column is reporting a keyword count under a verdict's
name.** Related: [[V-7]] is the same failure inverted, a search that cannot find
anything read as absence; this is a search that finds the repair and reads it as
the defect.

### M-18. A pre-registered kill-diagnostic whose value the corpus fixes, not the arm

`CEQ_V15_CONTRACT.md` PART IV, R1: *"Kill: bind passes, floor not crossed ⇒ `g`
unlearnable at budget — diagnose by linear probe on `log a` (should be
near-exact), never by a new construction."* The diagnostic is registered before
the data, which is right; what was never checked is whether it can return two
different answers.

**It cannot.** BED-M's coefficients are Rademacher
(`scale/negation_scope.py:428`):

```python
a = (torch.randint(0, 2, (n, s), generator=g).float() * 2 - 1).to(x.device)
a[:, :head + 1] = 0.0
```

so `a_i ∈ {−1, +1}` on the live band and `0` outside, giving `|a_i| ∈ {0, 1}`.
`log|a|` is therefore **identically 0 on the band** and `−∞` outside it. A linear
probe's `R²` on a constant target has `SST = 0.000e+00` and is undefined;
pooled and clamped (at `−30` or at `−100`, identically) it reads
`3.2466e-04` against a `d/N` null of `1.22e-04`.

The registered diagnostic returns NaN or `~0.0003` **whatever the arm does**.
Reading that as "`g` is unlearnable" would attribute to the architecture a
property of the corpus, and reading it as "`g` is learnable" is equally
unavailable. It is a kill condition that cannot fire and cannot fail to fire.

**The mechanism.** A diagnostic is registered against a QUANTITY (`log a`)
without checking the quantity's DISTRIBUTION in the corpus it will be run on. The
same probe on a corpus with continuous gates would be perfectly discriminating;
here the label's entire content lives in `sign(a_i)`, and `log|a|` is the one
function of `a` that throws that content away. This is `M-5` (a process that
cannot cross its own threshold) applied to a diagnostic rather than to a verdict,
and it is `V-8` (the PASS half's label is constant) with the constancy in the
regressand instead of the class label.

**The finding underneath it, which is larger than the diagnostic.** The contract
parametrizes the gate as `g = −softplus(W x)`, which is **monotone** in the drive
channel. The true `g = log|a|` is a band mask — **even** in the drive channel —
and `scale/m3_capability.py` gives its arms no positional feature to route
around it. Adding one squared feature recovers the band at `R² = 1.000000`
exactly. **The obstruction is evenness, not information**, and no amount of data
or optimization fixes a parametrization that cannot represent the target.

**Check.** Before a diagnostic is registered, run it on the CORPUS ALONE with no
arm and report its value and its null. A diagnostic whose no-arm value equals its
expected with-arm value is measuring the corpus. Concretely, for any probe
registered on a target `z`: print `Var(z)` over the draws it will see. `Var(z) =
0` is a blocked registration. Amendment for R1: probe `sign(a_i)` off the arm's
gate and report accuracy `p`, which is discriminating where `log|a|` is not.

### M-19. A dynamical invariant estimated on a float64 orbit that has already collapsed

The contract's S-G section carries `[RUN: 0.0003 at the right guard, 0.156 at a
wrong one]` for the Pesin deficit `λ̂ − h_sym`. Reproducing it, a probe first
returned `h_sym = 0` **at the generating partition** — the one place the deficit
is supposed to vanish for the right reason, returning the right answer for the
wrong one.

The cause is the arithmetic, not the estimator. A tent-map orbit in float64
loses one bit of the initial condition per step, so after roughly 52 iterations
the trajectory carries no information from `x₀` and collapses to a fixed point
of the rounding. Every symbol after that is an artifact of the last representable
bit. The instrument had to be rebuilt on exact rational arithmetic (`x = s/q`
with `q` prime) before it measured dynamics at all.

**Consequence for anything downstream.** Any `X33` deficit, any symbolic entropy,
any Lyapunov exponent computed by iterating a chaotic map in float64 past ~52
steps is measuring round-off. The contract's own `0.0003` and `0.156` inherit
this and are `INHERITED` under L-TIME until re-run in exact arithmetic — the
number may be right, but the run that produced it has not been shown to be.

**A second constraint on the same instrument, from the source.** Bollt et al.
(2001) prove the deficit is **non-monotone** in partition misplacement. Ranking
candidate guards by deficit is therefore unsound; only the "`≈ 0`" test is
admissible. A guard search that picks the argmin of the deficit is using the
instrument outside its stated behaviour.

**Check.** Any orbit-based invariant states its arithmetic and its horizon. In
floating point, the usable horizon is `mantissa_bits / log2(stretching rate)` —
about 52 steps for a tent map in float64 — and a run longer than that reports
round-off with a physical-looking name. Either use exact arithmetic, or state the
horizon and stop before it, or shadow the orbit and prove the shadowing.

## The eleven checks, before any control ships

Condensed from the above; this is the list to run down.

1. **Draw the instance, never hand-build it** (V-2). Report the count of draws on
   which it discriminates, and require it bounded away from 0 and n.
2. **Check the assertion is not an algebraic identity of your own construction**
   (V-3). Ask what set each side of the comparison ranges over.
3. **Check the branch under test actually executes** on your fixture (V-6).
4. **Check the PASS half's label is non-degenerate** (V-8): `sd > 0`,
   `0 < frac < 1`, both classes non-empty, discards counted.
5. **A reported absence needs a planted positive** (V-7, V-13, V-14) on
   identical instances, identical features, identical split — and entering at
   the instrument's **front door**, so it traverses every stage the real input
   does. Identical *text* is not identical *path*: a plant handed straight to
   the matcher cannot see a selector that selected nothing (V-14). Name the
   stages the plant passes through, and check the count of things examined as
   well as the count of things found.
6. **A repair must be shown to change the object it repairs** (V-9). Delete it
   in-process and watch the number move.
7. **Compute the control's expected value before it runs** (V-10, M-5). Print
   the ceiling arithmetic before the first number — and in the inference domain
   the same check is the design's finest achievable p against the α you intend
   to quote (M-9). A test that cannot reach its own α has already failed.

8. **A rule that CONDEMNS needs a planted negative** (V-15) — an instance
   carrying the proxy and innocent of the property. Rule 5's planted positive
   proves an instrument *can* fire; only a planted negative proves it fires on
   the right thing. Ask which direction your rule errs in, and plant against that
   direction.
9. **An instrument that cannot measure must say so** (V-16). Give it a planted
   *unreadable* — a missing tool, an empty response, a timeout — and confirm it
   reports UNKNOWN rather than a pass. Check that the unknown value is not the
   same value as the passing one, and that retrying does not launder it.

10. **A threshold travels only if its quantity is anchored** (V-17). Before
   importing one, ask what value the quantity takes under a null or trivial
   predictor. A construction-independent constant (NRMSE's 1.0) travels; a ratio
   calibrated against a specific generator does not. Say which quantity the source
   measured, not just where the number came from.

11. **A reading that bypasses the shared reader is not guarded by it** (V-18).
   For every analysis that consumes a journal, check it CALLS the module that
   deduplicates and refuses, rather than regrouping the rows inline. The guard
   constrains its callers and nothing else.

And one more that costs more than all eleven when it is skipped: **state the
regime in which your baseline is optimal, and check your task is not in it**
(D-1).
