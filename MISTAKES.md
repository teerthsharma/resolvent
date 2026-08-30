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
no `--task` flag"*; `--task` was added at `scale/m3_quintuple.py:617` (FINDINGS
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
(`CHECKLIST.md:1168`, `ceq/hf_artifact/README.md:35`).

**Half the repair is already in the tree, which is what makes the rule
concrete.** Both of those rows already print the seed-agreement count beside the
interval — `ceq/hf_artifact/README.md:35` reads
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
file that ships (`ceq/hf_artifact/README.md:80`), naming both families and both
endpoints. The table three lines above it still prints `95% CI` with no
estimator, and the JSON behind that table **already carries `n_boot: 10000` and
`boot_seed: 0`** — the provenance exists and is dropped at render time. A caveat
that is correct, complete, in the right place and below the row that gets quoted
is the exact failure P-8 describes, and this is it happening to the
repository's most-quoted result while P-8 was being written.

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
`self.readout(h).squeeze(-1)[:, s - 1]`, `scale/m3_quintuple.py:311` (FINDINGS
C1). One softmax layer is **provably Bayes-optimal on exactly that shape**,
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

## The seven checks, before any control ships

Condensed from the above; this is the list to run down.

1. **Draw the instance, never hand-build it** (V-2). Report the count of draws on
   which it discriminates, and require it bounded away from 0 and n.
2. **Check the assertion is not an algebraic identity of your own construction**
   (V-3). Ask what set each side of the comparison ranges over.
3. **Check the branch under test actually executes** on your fixture (V-6).
4. **Check the PASS half's label is non-degenerate** (V-8): `sd > 0`,
   `0 < frac < 1`, both classes non-empty, discards counted.
5. **A reported absence needs a planted positive** (V-7) on identical instances,
   identical features, identical split.
6. **A repair must be shown to change the object it repairs** (V-9). Delete it
   in-process and watch the number move.
7. **Compute the control's expected value before it runs** (V-10, M-5). Print
   the ceiling arithmetic before the first number — and in the inference domain
   the same check is the design's finest achievable p against the α you intend
   to quote (M-9). A test that cannot reach its own α has already failed.

And one more that costs more than all seven when it is skipped: **state the
regime in which your baseline is optimal, and check your task is not in it**
(D-1).
