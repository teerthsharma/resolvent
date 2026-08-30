# AUDIT.md &mdash; the P0.1 census

Round 10, PHASE 0, iteration 1. Seat: SATURN (instruments / controls / manifests / MISTAKES.md).
Branch `feat/r9-causal-consequence`. One row per test file, per scale module, per root doc.

**Amended at iteration 1, before shipping, by an attack from MARS that FIRED.** Presumption P1 as
issued in the brief was replaced by P1' and the sheet re-derived. The replacement and MARS's numbers
are recorded in full under *Presumptions* below. Nothing under `.claude/worktrees/` is in scope; it is
excluded from every search here, and `tests/deimos/test_deimos_r9_iteration1.py:266` records the same
exclusion being got wrong before (MISTAKES.md V-13, a walk that included nested checkouts).

## Counts, with the command that produced each

```
git ls-files -- 'tests/*.py'  | wc -l   ->  191   test files
git ls-files -- 'scale/*.py'  | wc -l   ->   99   scale modules
git ls-files -- '*.md' | grep -v / | wc -l -> 53   root docs
                                        +    4   round reports misfiled under tests/
                                        = 347   AUDITED ROWS
git ls-files -- 'tests/' | wc -l        ->  197   (NOT the row count - see the layout defect below)
git ls-files -- 'results/' | wc -l      ->  242   tracked files, of which only 27 are .jsonl
ls results/*.jsonl | wc -l              ->   27   journals actually queryable
```

**Layout defect, P0.4.** `git ls-files -- 'tests/'` returns 197, but 6 of those are not test files.
Four are agent round reports and two are journals. A naive test-tree count therefore **over-reports by 6**,
and a naive root-doc count **under-reports by 4**. This matters past bookkeeping: iteration 2's spot-check
is a binomial draw whose denominator is this sheet's, and 6 undrawable rows in the urn corrupt the draw.

**Denominator correction, P0.4.** The brief describes the journal as `results/*.jsonl (242 tracked files)`.
Measured, those are two different sets: 242 is the whole `results/` tree across five extensions
(165 `.pt`, 37 `.txt`, 27 `.jsonl`, 7 `.json`, 6 `.md`), and **27** are journals. Every `last-run state`
cell below is a query against those 27, holding 809 records of which 391 carry the `key`/`meta`/`value` shape.

### journals found outside results/

Per L-G2 journals never move and are not audited as test files. These two are listed, not classed.
Their location is itself the P0.4 finding: a journal under `tests/` is invisible to every instrument
that globs `results/*.jsonl`, which is all of them.

| path | bytes | records |
|---|---|---|
| `tests/chase/deq_run.jsonl` | 186958 | 970 |
| `tests/chase/scale_axes.jsonl` | 1823 | 3 |

## Census result

| class | rows |
|---|---|
| live | 299 |
| vacuous | 8 |
| orphan | 25 |
| superseded | 14 |
| struck | 1 |
| **total** | **347** |

| disposition | rows |
|---|---|
| KEEP | 314 |
| ATTIC | 33 |

**Presumed rows: 36.** A presumption is not a verdict. Every one is marked in its reason cell and is a
spot-check candidate for iteration 2. **NO JOURNAL ENTRY: 198 of 347** &mdash; a real state, not a blank.

## Presumptions, as applied

### P1 &mdash; REPLACED AT ITERATION 1. Do not apply it.

The brief issued: *a control that constructs its own input = vacuous-by-scope until shown otherwise.*

MARS attacked it and the attack fired. His case, which I re-derived in-tree before accepting it:
`tests/cameron/test_impact_hetero_is_not_its_own_baseline.py` draws every tensor it asserts on through
`NS.M3_TASKS[name]` &mdash; verified here at `:62-63`, where the `drawn` fixture asserts `name in NS.M3_TASKS`
and returns the registry entry. Under P1 that reads as *constructs its own input* and would be ATTIC'd.
MARS restored the V-1 defect at `scale/negation_scope.py:1351` and re-ran: **shipped registry 6 passed,
defect restored 4 failed / 2 passed**. The control has a rejection region, and P1 would have attic'd the
live instrument binding MISTAKES.md class V-1. (Those two numbers are MARS's; I did not re-run them.
What I verified myself is the front-door draw at `:62-63` and the six test functions at `:79, 96, 125, 135, 151, 167`.)

The mechanism is the part that generalises. **P1 keys on WHO built the input. The repo's own rule keys on
WHAT PATH the input takes** &mdash; MISTAKES.md V-14: a planted positive must enter at the instrument's front
door and traverse every stage the real input traverses. Every planted positive is constructed; that is what
planting means. So P1 has no discriminating power and fires on cures as readily as on diseases.

### P1' &mdash; the rule actually applied to this sheet

> A control is VACUOUS-BY-SCOPE when its planted input enters the instrument **below its front door**, so
> that one or more stages the real input traverses go unexercised. Name the skipped stages. A plant that
> enters through the same object production reads is not vacuous by scope and must be classed **by measurement**.

P1' has a demonstrated rejection region, checked because a rule that never fires is its own defect. It fires
on `scale/chase_struck_coverage.py`, verified here: `control()` at `:105` calls `scan_text()` on two literal
strings at `:107` and `:109` and **never calls `collect_targets()`**, which is reached only at `:137` and
walks the repository. The control exercises the matcher and never the reach &mdash; MISTAKES.md V-14 exactly.

Measured application, `python scratchpad/p1prime.py`:

```
test .py files                                              191
files reaching a production builder (front door)             46
files with NO front-door call                               145
of the 27 rows P1 would have condemned, P1' still fires on   27
of the 27 rows P1 would have condemned, P1' clears            0
```

My front-door bucket is **46**; MARS measured **22**. Both are measurements and the gap is the builder
vocabulary, not a disagreement about any row: mine additionally counts `contrast(`, `Journal(`,
`read_journal`, `SHIPPED_CASE` as front doors. The 22 is the stricter reading. Recorded rather than
reconciled, because reconciling it by picking the number I prefer is how a denominator gets chosen to suit.

### P2 &mdash; a symbol absent from every reference = orphan
### P3 &mdash; zero importers AND zero results = orphan

P3 **produced two false positives and was corrected before shipping**; see FINDING 1. Its corrected form
resolves `Journal(NAME)` to `results/{NAME}.jsonl` and counts prose citations, not just python imports.

## FINDING 1 (test-bound, RED first) &mdash; a literal-path census under-reports journal producers

`scale/bucket.py:45` builds `self.path = RESULTS / f"{name}.jsonl"`. The journal path is **constructed**
and never written down in the producing module. A census that resolves producers by matching the literal
string `results/<name>.jsonl` therefore reports NO JOURNAL ENTRY for live producers, and on rule P3
(*zero importers AND zero results = orphan*) presumes them dead.

It did. My own first pass classed `scale/s2_units.py` ORPHAN/ATTIC. It is the producer of
`results/s2.jsonl` (13 records), and **13 of its 15 six-significant-digit readings reproduce from
`results/*.jsonl` at abs=5e-7**. `scale/r2_units.py` produces `results/r2.jsonl` (36 records) and was
classed the same way. Both are corrected on this sheet.

This is MISTAKES.md **V-7** &mdash; a search structurally incapable of finding a thing, its silence read as
absence &mdash; which is the exact defect `scale/journal_scan.py` was written to abolish. The instrument
existed; the failure was reintroduced one level up, in the code that decides which files to hand it.

**RED, against the tree as it stands:**

```
$ python -m pytest tests/saturn/test_journal_path_is_discoverable.py -q
3 failed, 3 passed in 0.23s
FAILED ...::test_the_journal_a_module_writes_is_named_in_that_module[scale/m2_units.py-m2]
FAILED ...::test_the_journal_a_module_writes_is_named_in_that_module[scale/r2_units.py-r2]
FAILED ...::test_the_journal_a_module_writes_is_named_in_that_module[scale/s2_units.py-s2]
```

The 3 passing are its two must-fire arms plus the premise bind: the file asserts that `bucket.py` still
constructs the path (else the finding is moot) and that at least one module calls `Journal(NAME)` (else the
parametrised assertion ranges over an empty set and is itself a vacuous control).

## FINDING 2 (test-bound, RED first) &mdash; P1' misfires on refutation instruments

A **refutation instrument** exists to REFUTE a claim: it defines `test_claim_*` functions that assert the
optimistic statement, so that **its failing IS the result**. The convention is the repo's own &mdash;
`DONE_ARCHIVE_ROUND1.md:1013` lists exactly these names under a leading `RED:`.

`grep -rln "def test_claim_" tests/` finds **12** such files. **The first pass of this census ATTIC'd 5 of them**
as vacuous-by-scope: `test_operator_is_a_triangular_solve.py`, `test_ppnp_parity.py`, `test_sheaf_kernel_parity.py`,
`test_signedness_is_not_new.py`, `test_topology_washout.py`. All five are corrected to live/KEEP on this sheet.

The sharpest case is the one that would have cost most. `tests/foreman/test_signedness_is_not_new.py` asks
*"is SIGNEDNESS alone enough to be new?"* and reimplements **SimA** (Koohpayegani & Pirsiavash, arXiv 2206.08898)
and **Differential Transformer** (Ye et al., arXiv 2410.05258, ICLR 2025) **from their defining equations**, racing
them on this module's own instrument. It is the prior-art defence of the contribution. P1' condemned it.

**Why P1' misfires, which is the part that generalises.** A refutation instrument constructs its own input
**by necessity**: the mechanism it races is a competitor's, and a competitor has no production batch path in
this repository. Building SimA from its paper is the entire method, not a shortcut around one. P1' keys on
*where the input enters*; for these files the governing scope is **the claim**, not the batch.

That is now **three presumptions in this one census that failed by keying on a surface feature rather than on
the instrument's role** &mdash; P1 (who built the input; killed by MARS), P3 (a literal path; killed by FINDING 1),
and P1' here. The pattern is the finding: a census rule that reads syntax will keep mistaking method for defect.

### P1'' &mdash; the carve-out now applied

> P1' does NOT apply to an instrument whose RED is its deliverable. Identify one by the repo's own convention:
> it defines `test_claim_*` functions, or it reimplements a cited external mechanism from that source's defining
> equations. Such a file constructs its input by necessity and is classed by its ROLE, never by input origin.

**RED, against `AUDIT.md` as it stood before this correction:**

```
$ python -m pytest tests/saturn/test_census_does_not_attic_refutation_instruments.py -q
5 failed, 9 passed in 0.95s
FAILED ...::test_a_refutation_instrument_is_never_atticked[tests/foreman/test_operator_is_a_triangular_solve.py]
FAILED ...::test_a_refutation_instrument_is_never_atticked[tests/foreman/test_ppnp_parity.py]
FAILED ...::test_a_refutation_instrument_is_never_atticked[tests/foreman/test_sheaf_kernel_parity.py]
FAILED ...::test_a_refutation_instrument_is_never_atticked[tests/foreman/test_signedness_is_not_new.py]
FAILED ...::test_a_refutation_instrument_is_never_atticked[tests/foreman/test_topology_washout.py]
```

The 9 passing are the 7 correctly-classed refutation instruments plus two must-fire arms (the sheet parses to
>=300 rows; at least one `test_claim_*` file exists). It goes GREEN once the sheet above is corrected, and then
stands as the regression guard that stops the next census re-making the same mistake.

## FINDING 3 (measured, not presumed) &mdash; a RED withdraws the vacuity presumption

22 of the 27 files P1' condemned were RUN (5 do not collect as a set). Measured, one command:

```
$ python -m pytest <the 22 files> -q --tb=no -rf
61 failed, 197 passed, 34 xfailed, 15 warnings in 372.16s (0:06:12)

files carrying >=1 measured RED   12 of 22
failures in test_claim_* names    22 of 61   (exactly the 5 files corrected by FINDING 2)
failures under other names        39 of 61   (7 further files, corrected here)
```

**A test that is currently rejecting is not passing by construction, and therefore cannot be vacuous in the
strong sense.** P1' is worded *vacuous-by-scope UNTIL SHOWN OTHERWISE*; a demonstrated live rejection region
shows otherwise on the only question a census can settle cheaply. All 12 move to live/KEEP.

Their docstrings confirm it independently &mdash; each is bound to a pre-registered falsifier or a named control:

- `tests/foreman/test_r2_maxplus_reduction.py` (10 reds) &mdash; REQUIREMENTS.md R2 names APPNP as *"one control
  the module must beat"*.
- `tests/foreman/test_r1_settling.py` (8 reds) &mdash; quotes the R1 falsifier verbatim, including *"If pass 1
  already equals the settled state, R1 is dead weight -- delete it."*
- `tests/chase/test_scale_axes.py` (10 reds) &mdash; *"Does the 1.0334 parity ratio survive to 300M, or is it a
  3.3M artifact?"*
- `tests/cameron/test_schedule_ablation.py` (3 reds) &mdash; THEORY.md sec 3's *"control to beat"*, measured
  **on REAL attention from a pretrained model**. That last clause means P1' misread this file outright: its
  input IS a production artifact, and the input-origin regex simply got it wrong.

**What a RED does NOT settle.** It does not prove the SCOPE is right &mdash; a test can fail for a reason
unrelated to the claim it names. The vacuity question is narrowed, not closed, and these rows are exactly
what iteration 2 should spot-check. The 10 files that ran ALL-GREEN keep their presumption, because passing
is precisely what a vacuous control does and green is not evidence either way.

### Routed, not priced &mdash; SATURN does not price and does not adjudicate

Three measured results below belong to the pricing and adjudication seats, recorded here only because the
census run surfaced them:

1. `test_the_stated_R1_falsifier_rejects_the_affine_update_it_is_meant_to_kill` **FAILS**. A must-fire arm
   that is not firing: on this checkout the R1 falsifier does not reject the update it exists to kill. By
   MISTAKES.md V-10 that is a live vacuity finding **about an instrument the module rests on**, not about a test.
2. `tests/foreman/test_r2_maxplus_reduction.py` fails **10** assertions against APPNP, the one control
   REQUIREMENTS.md R2 says the module must beat &mdash; including
   `test_maxplus_star_differs_from_appnp_in_appnps_own_parameterization`.
3. `tests/chase/test_scale_axes.py` fails **10**, among them
   `test_the_parity_ratio_does_not_degrade_with_sequence_length` and `test_the_tuned_rho_is_the_same_at_two_scales`.

Whether these are recorded refutations already priced in an earlier round, or regressions on this branch, is
not a census question and is not answered here. They are handed over as measured state.

## Iteration 2 &mdash; the spot-check, and what it moved

The draw was taken out of the census author's hands: `scale/spotcheck_draw.py`, SEED=10002, output
`results/r10_it2_draw.txt`, fixed and committed before any row was measured. **The draw reproduces
against `AUDIT.md@06a180c`** &mdash; the sheet as it stood before the amendments below &mdash; and not
against this file as it now reads, because the amendments moved KEEP 307/ATTIC 40 to 314/33. That is
the correct relation and it is recorded rather than papered over: a draw must reproduce against the
urn it was drawn from. `git show 06a180c:AUDIT.md` restores that urn.

The protocol defining what "passes" and "is dead" mean per row type was written and committed at
`06a180c`, `results/r10_it2_protocol.md`, **before** any drawn row was run. Three root docs landed in
the KEEP sample and "must pass" is undefined for a document until it is defined.

**Verdict: 1 KEEP failure of 10. Under the pre-registered binomial (P(>=1 fails | sheet >=95% right)
= 0.4013), one failure AMENDS the row and the census STANDS.** The failure is `workdone2.md`.

### What the spot-check found, all four test-bound in `tests/saturn/test_r10_it2_spotcheck_reds.py`

1. **The "no python importer" cell is an absence claim from a search that could not find what it was
   looking for.** 18 rows assert it; **12 (67%) are wrong**, missing **135 front-door importer edges**
   &mdash; `scale/pivot_probe.py` has 52 and `scale/negation_scope.py` 45, both recorded as having
   none. This is MISTAKES.md V-7 again, and it is the same defect as FINDING 1 one level up.
   **It cost zero dispositions**: all 12 are live/KEEP already, carried by their journal leg, and the
   KILL 3 orphan class survives its re-measure intact &mdash; **0 of 19 scale orphans have a
   front-door importer**. A large defect that changed no verdict is still a defect, and it is the one
   most likely to change a verdict on the next sheet that lacks the second leg.

2. **Seven of the twelve files named on `tests/chase/conftest.py`'s KNOWN_RED ledger were classed
   ATTIC.** The ledger resolves each key against collected node ids and applies `xfail(strict=True)`
   at every collection: that is a live reader holding a live reference, and each entry carries a
   measured finding about the shipped deliverable. The census read them as *"ran ALL-GREEN in the
   iteration-1 sweep"*. A file whose every test is a strict xfail reports exit 0 and zero failures.
   **`tests/chase/test_hf_shipping.py`, the drawn ATTIC row, ran 5 xfailed and 0 passed** &mdash;
   ALL-GREEN was read off an exit status that cannot tell a passing test from a recorded finding.
   All seven are amended to live/KEEP.

   **This is the fourth presumption in this census to fail by keying on a surface feature rather than
   on the instrument's role** &mdash; P1 (who built the input), P3 (a literal path string), P1' (where
   the input enters), and now the vacuity sweep (an exit status). The pattern was predicted at
   FINDING 2 and it recurred, which makes it the census's most reliable finding about itself.

3. **`workdone2.md`'s retention ground is empty.** Its cell read *1/10 readings reproduce at
   abs=5e-7*; re-measured it is **0 of 4**. The extractor is calibrated against this sheet's own
   control &mdash; `LOOP_PROMPT_ROUND6_ARCHIVE.md` re-measures **8/23, exactly its cell** &mdash; so
   the disagreement is the sheet's, not the instrument's. The nearest journal values miss by 1.6e-4 to
   35.5, between 300x and 7e7x the tolerance: near-misses, not a parsing artifact. **KEEP stands on a
   different, measured ground** (7 tracked files name it), and the row is amended to say so. It is
   kept as a cited round log, not as provenance.

4. **Three files in `tests/chase` declare "RED on purpose" and are absent from the KNOWN_RED ledger**
   &mdash; `test_structural_zero_guard.py` (the drawn row), `test_checklist_kills_are_evaluable.py`,
   `test_m2_verdict_nan.py`. `tests/chase` is the one directory carrying a ledger, so inside it an
   intentional red belongs there; `tests/foreman` reds are unledgered by design because foreman has no
   ledger. These three surface as hard CI failures indistinguishable from regressions.
   **Not fixed here.** Adding ledger entries changes what CI treats as an expected failure, which is
   CHASE's seat and not the census author's. Routed, left RED as the deliverable.

### The five ATTIC rows, re-measured rather than re-asserted

`tests/cameron/test_hankel_mustfire.py` (16 passed; no front-door call, no `test_claim_*`, no external
mechanism &mdash; conjunction holds), `titan-report.md` (0 tracked files name it or its stem under the
protocol's exclusions), `scale/orbit_probe.py` (0 importers, 0 `results/` writes at any extension, 0
prose citations), `scale/sparse_probe.py` (0 importers, 0 `results/` writes; prose citations amended
4 -> 3 under the exclusions) are **DEAD, confirmed**. `tests/chase/test_hf_shipping.py` is **NOT
dead** &mdash; finding 2 above.

### Power

Whether a uniform draw of 10 is powered against CONCENTRATED error is MARS's parallel calculation and
is **not claimed here**. Finding 2 is exactly a concentrated error &mdash; 7 rows in one directory,
one class, one presumption &mdash; and the uniform draw caught it only because `test_hf_shipping.py`
happened to be drawn. That is luck, not power.

## The kills, and the route that replaces each

Every ATTIC of something that was doing real work is a kill and ships a replacement route.

### KILL 1 &mdash; the impact / impact_hetero TASK (pre-seeded ATTIC)

Carried verbatim. **The repaired code stays** &mdash; `scale/impact.py` and `scale/impact_hetero.py` are KEEP on
this sheet, because no journal says the code is dead; only the task is inadmissible.

- **What it was FOR:** a corpus where a planted cause propagates to a measurable consequence, so an arm can
  be scored on consequence rather than resemblance.
- **Route (reroute):** the E-task / C1-propagate families already registered in `scale/negation_scope.py`
  `M3_TASKS`, journalled at `results/m3_quintuple_v2.jsonl` under keys `settled_k8_..._taske3_t{1,2,8,32}`
  and `..._taskc1_propagate_t{1,2,8,32}` &mdash; 151 records present.
- **Why sharper, one line:** the label is an equilibrium the arm must reach rather than a linear function of
  the input, so the 4.93e-08-train / 1.478-eval memorisation gap that killed impact cannot recur by construction.

### KILL 2 &mdash; 15 test files presumed vacuous-by-scope under P1'

- **What they were FOR:** binding a claim about the shipped operator to a runnable assertion.
- **Route (reprice, not retire):** each names the stages its plant skips; the repair is to move the plant up
  to the front door &mdash; draw through `NS.M3_TASKS[...]` or the arm's own builder rather than a local
  `torch.randn`. `tests/cameron/test_impact_hetero_is_not_its_own_baseline.py:62-63` is the worked example of
  the shape that passes P1', and it is 2 lines: assert the key is registered, return the registry entry.
- **Why sharper, one line:** a plant entering at the front door exercises every stage between the door and
  the assertion, so the same test also covers the binding that `M3_TASKS` never validates on import.
- **Status: PRESUMED.** These are the rows iteration 2's binomial most needs to cover.

### KILL 3 &mdash; scale modules and test helpers presumed orphan

- **What they were FOR:** one-shot probes answering a question a round asked.
- **Route (retire, with the measured fact):** the question each answered is closed in the round log that
  cites it &mdash; the reason cell names the citing file where one exists. A probe with zero importers, zero
  journal records and zero prose citations has no reader; retiring it removes a file, not a result.
- **Why sharper, one line:** what these measured, where it survived at all, survives as a journalled record
  in `results/`, which outlives the script that wrote it.
- **Caution, and it is the whole reason this class is PRESUMED:** FINDING 1 shows this exact rule producing
  false positives on constructed paths. Do not act on a KILL 3 row without the spot-check.

**Verification run on this class before shipping,** because FINDING 1 had already burned it once. Every one of
the 18 scale rows in KILL 3 was re-checked for a `results/` write of ANY extension, not just `.jsonl`:

```
for f in <the 18 KILL-3 scale modules>; do grep -o 'results/[A-Za-z0-9_./-]*' "$f"; done
  -> exactly one hit, scale/run_m2.py:57  pathlib.Path("results/m2.json").write_text(...)
git ls-files results/m2.json  -> (empty)      ls results/m2.json -> No such file or directory
```

So `scale/run_m2.py` is a producer whose output was **never committed**, and the other 17 write no `results/`
path at any extension. The orphan class survives its own re-check on this bucket. That is a re-check, not a
spot-check: it rules out the `.json`/`.txt` blind spot, not the `Journal(NAME)` one, which is why these rows
stay PRESUMED.

### KILL 4 &mdash; root docs and in-tree reports named by no other tracked file

- **What they were FOR:** an architectural decision record and per-round agent reports.
- **Route (reroute):** `ADR-001-bucketed-measurement.md` is uncited by name, but the decision it records is
  **live and load-bearing** &mdash; `scale/bucket.py` implements it and its docstring cites ADR-001 by name.
  The route is to cite the ADR from `bucket.py` by filename, not to move the file. This row is an artifact of
  citation-by-stem and is the single clearest illustration of why P2 needs the spot-check.
- **Why sharper, one line:** a one-line citation makes the reference measurable, where atticking the ADR
  would delete the reasoning `bucket.py` exists to implement.

## Open &mdash; convincing, not yet test-bound. Not admissible as findings.

1. **Five of the 27 P1'-condemned files do not collect together.** `tests/chase/test_kernel_contracts.py`,
   `test_multizoom_cost.py`, `test_multizoom_kernel.py`, `test_multizoom_r5.py`,
   `test_rollback_flex_attention.py` produce 5 collection errors when run as a set, while
   `test_kernel_contracts.py` collects clean **alone**. That signature is a module-name collision or an
   import-order interaction, not a broken test. Their `last-run state` cells are honest about this. No RED,
   so it stays here.
2. **160 of 242 tracked `results/` artifacts are named by no tracked file** (`python` scan over all tracked
   text). Most are `.pt` weights under `m3_quintuple_v2_cuda_weights/`, which a manifest may legitimately
   reach by glob rather than by name. Not a finding until someone measures whether the manifest enumerates them.
3. **`scale/vgpe_flops.py` is a producer with no journal.** `V12_PRICING.md:3` names it *the* producer
   (`Run it with python scale/vgpe_flops.py`) and rests 377 lines on its output, but it writes no `results/`
   artifact, so nothing replays it. Priced as a P-1 risk (a number whose only home is prose), not asserted:
   I have not checked whether the doc's figures reproduce.
4. **`titan-report.md` is uncited and self-referential** &mdash; its own `:4` points at a copy under
   `.superpowers/`, which is out of scope. Classed orphan on the measured rule, but it is 4 days old and may
   simply be new. Flagged rather than argued.

## The STRUCK registry &mdash; where it actually lives

**There is no `STRUCK.md` at root.** The registry is `tests/loop/test_no_struck_constant_ships.py`, in the
module-level `STRUCK: dict[float, str]` at `:47`, mapping each struck value to its provenance. `inspector.py:459`
carries the control `"struck registry is non-empty"`. Cross-references that resolve to it:
`tests/cameron/test_harmonic_attribution.py:22` and `:182`, `CHECKLIST.md:478` and `:486`, `D1.md:215`,
`PREREGISTRATION_HOLE_AUDIT.md:439`, `scale/equilibrium_probe.py:130`.

## The 11/11 AttributeError family &mdash; exact size, measured

**One file. Eleven tests.** `tests/cameron/test_harmonic_attribution.py`, 411 lines on disk, of which the
302 named in its own docstring are the record kept verbatim. Eleven `def test_` at `:199, 218, 233, 249, 266,
280, 298, 320, 337, 355, 362`. The skip is module-level at `:176` with `allow_module_level=True`, guarded by a
live `hasattr` check at `:170-174` over nine names on `scale/negation_scope.py`, so it un-skips by itself the
day a producer is written. It is a skip and **not** an `xfail(strict=True)` on purpose: a strict xfail would
file these eleven alongside the ~50 at `tests/chase/conftest.py:154` that ARE the record reproducing.

`grep -rl "allow_module_level" tests/ | wc -l` returns **2**; the other is the file asserting this one's
absence. The family is therefore exactly one file, not a family of many, and the brief's phrasing
("every test in the 11/11-AttributeError family") resolves to 11 test functions in 1 file.

## The rows

### tests/*.py

Count: **191** &mdash; `git ls-files -- 'tests/*.py'`

| path | claims to test | last-run state FROM THE JOURNAL | class | KEEP/ATTIC | one-line reason |
|---|---|---|---|---|---|
| `tests/cameron/conftest.py` | --------------------------------------------------------------- board logging | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/cameron/dilated.py` | A banded sgate whose band may be DILATED, plus the composed sign-flip probe. | NO JOURNAL ENTRY | live | KEEP | helper imported by 5 python module(s) |
| `tests/cameron/interventional.py` | Interventional corpora with oracle-measured outcomes. | NO JOURNAL ENTRY | live | KEEP | helper imported by 2 python module(s) |
| `tests/cameron/ladder.py` | One CLI so every ladder cell is produced by the SAME code path. | 1/1 readings reproduce at abs=5e-7 | live | KEEP | helper imported by 2 python module(s) |
| `tests/cameron/perron.py` | A nonnegative, Perron-certified settling operator. CPU and CUDA, float64. | NO JOURNAL ENTRY | live | KEEP | helper imported by 7 python module(s) |
| `tests/cameron/schedule_sweep.py` | Does a NON-GEOMETRIC dilation schedule repair the severance? | NO JOURNAL ENTRY | live | KEEP | helper imported by 1 python module(s) |
| `tests/cameron/severed_fraction.py` | THE SEVERED FRACTION -- the measurement Cameron named and could not run. | 1/2 readings reproduce at abs=5e-7 | orphan | ATTIC | helper: 0 test functions, 0 python importers, no journal; see KILL 3 |
| `tests/cameron/test_arc_reality.py` | ARC-AGI as a MEASURING INSTRUMENT, before ARC-AGI as a target. | NO JOURNAL ENTRY | vacuous | ATTIC | PRESUMED vacuous-by-scope (L-SCOPE): builds its own tensor/graph instead of drawing from the production batch path, and 0 of its 0 6-significant-digit readings reproduce from any results/*.jsonl at abs=5e-7. ran ALL-GREEN in the iteration-1 sweep, which is exactly what a vacuous control does and is therefore not evidence either way. PRESUMPTION, NOT VERDICT - iteration-2 spot-check candidate; see KILL 2 |
| `tests/cameron/test_bar_control_sees_the_arms_preprocessing.py` | A7: the bar's trained control and the arms must see the same label. | NO JOURNAL ENTRY | live | KEEP | 4 tests, input drawn from the scale/ production path |
| `tests/cameron/test_c1_propagate_registration.py` | C1 -- the VECTOR consequence corpus, and its admission bundle. | 5/21 readings reproduce at abs=5e-7 | live | KEEP | 8 tests, input drawn from the scale/ production path |
| `tests/cameron/test_capability_result.py` | The capability number, bound to the run that produced it. | NO JOURNAL ENTRY | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/cameron/test_cogs_harness.py` | COGS, wired. The blocker in `harness.run("cogs")` was a fixed 256-way head. | NO JOURNAL ENTRY | live | KEEP | 7 tests, input drawn from the scale/ production path |
| `tests/cameron/test_composition_is_the_uncosted_route.py` | The stated tradeoff, encoded as an assertion, on the shipped operator. | 2/7 readings reproduce at abs=5e-7 | live | KEEP | builds its own input, BUT 2/7 of its readings reproduce from the journal at abs=5e-7 - its scope is anchored to the production path, so the L-SCOPE presumption does not fire |
| `tests/cameron/test_diagnose_package.py` | The stranger-command: one line, CPU, no GPU, no network. | NO JOURNAL ENTRY | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/cameron/test_domain.py` | Is there an interventional domain that is ALREADY linguistic? | NO JOURNAL ENTRY | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/cameron/test_e4_harmonic.py` | The e3-harmonic ladder: its oracle, its two gates, and the band it cannot reach. | 1/6 readings reproduce at abs=5e-7 | live | KEEP | 10 tests, input drawn from the scale/ production path |
| `tests/cameron/test_e4_rips_gate.py` | E4 admission: the Rips connectivity label is struck, and the reroute that saves it. | NO JOURNAL ENTRY | live | KEEP | 8 tests, input drawn from the scale/ production path |
| `tests/cameron/test_e4prime_registration.py` | X19: the E4' task family (join the two largest components) in M3_TASKS. | NO JOURNAL ENTRY | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/cameron/test_hankel_mustfire.py` | Must-fire for the Hankel instrument, in both directions, on DRAWN instances. | NO JOURNAL ENTRY | vacuous | ATTIC | PRESUMED vacuous-by-scope (L-SCOPE): builds its own tensor/graph instead of drawing from the production batch path, and 0 of its 0 6-significant-digit readings reproduce from any results/*.jsonl at abs=5e-7. ran ALL-GREEN in the iteration-1 sweep, which is exactly what a vacuous control does and is therefore not evidence either way. PRESUMPTION, NOT VERDICT - iteration-2 spot-check candidate; see KILL 2 |
| `tests/cameron/test_hankel_worked_example.py` | K-5 calibration: reproduce the LOOP_PROMPT.md section 1.1 worked calculation. | 2/7 readings reproduce at abs=5e-7 | live | KEEP | builds its own input, BUT 2/7 of its readings reproduce from the journal at abs=5e-7 - its scope is anchored to the production path, so the L-SCOPE presumption does not fire |
| `tests/cameron/test_harmonic_attribution.py` | U1 / N3 -- HARMONIC-MEASURE ATTRIBUTION AND ITS CROSS-CHECK CONTROL. | NO JOURNAL ENTRY | struck | KEEP | module-level skip fires at collection: 9 names it calls on scale/negation_scope.py have zero defining commits across all refs. 302 lines kept verbatim per the 5.4944e-13 precedent; skip not xfail(strict) so these 11 are not filed with the ~50 that ARE the record |
| `tests/cameron/test_harness.py` | The eval harness: a number the day the model finishes training. | NO JOURNAL ENTRY | live | KEEP | 11 tests, input drawn from the scale/ production path |
| `tests/cameron/test_headline_ci_provenance.py` | The headline `settled - softmax` interval, pinned to the run that made it. | results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/cameron/test_identity_manifest.py` | X-R1 -- MEASURED-OBJECT != SHIPPED-OBJECT. The identity manifest. | results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | 9 tests, input drawn from the scale/ production path |
| `tests/cameron/test_impact_hetero_is_not_its_own_baseline.py` | A1: `impact_hetero` must draw a different corpus from `impact`. | NO JOURNAL ENTRY | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/cameron/test_m3_counter_squared.py` | `counter_squared` as a runnable task in the M3 corpus, and the gap after encoding. | 1/61 readings reproduce at abs=5e-7 | live | KEEP | 20 tests, input drawn from the scale/ production path |
| `tests/cameron/test_m3_etasks.py` | The E-task family: the label is an EQUILIBRIUM, not a static expression of x. | NO JOURNAL ENTRY | live | KEEP | 18 tests, input drawn from the scale/ production path |
| `tests/cameron/test_minimum_arch.py` | How few of the four things does the win condition actually need? | NO JOURNAL ENTRY | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/cameron/test_mistakes_citations_resolve.py` | MISTAKES.md must not rot into the failure classes it documents. | NO JOURNAL ENTRY | live | KEEP | 7 tests, input drawn from the scale/ production path |
| `tests/cameron/test_negation_is_the_axis.py` | Which benchmark could see this module's one distinguishing property? | NO JOURNAL ENTRY | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/cameron/test_parity_is_the_wrong_target.py` | Is val-loss parity on TinyStories bytes the right target at all? | 1/5 readings reproduce at abs=5e-7 | live | KEEP | 4 tests, input drawn from the scale/ production path |
| `tests/cameron/test_published_intervals_have_producers.py` | Both published interval families reproduce from the journal. | results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/cameron/test_r3_perturbation.py` | R3 -- amplify few, crush many. VERDICT: DELETE. | 1/1 readings reproduce at abs=5e-7 | live | KEEP | 7 tests, input drawn from the scale/ production path |
| `tests/cameron/test_r4_compression.py` | R4 -- compress by forgetting. VERDICT: DELETE. | 1/13 readings reproduce at abs=5e-7 | live | KEEP | 7 tests, input drawn from the scale/ production path |
| `tests/cameron/test_r5_aggregator_red.py` | CAMERON round 5 -- the RED tests behind the aggregator finding. | results/arm_a.jsonl (3 rec); results/cameron_aggregators.jsonl (8 rec) | live | KEEP | 9 tests, input drawn from the scale/ production path |
| `tests/cameron/test_r6_agreement.py` | R6 -- local readings, global agreement. VERDICT: DELETE. | NO JOURNAL ENTRY | live | KEEP | MEASURED RED, presumption WITHDRAWN: this file has a demonstrated live rejection region - it is currently rejecting, so it is not passing by construction and cannot be vacuous in the strong sense. Its docstring binds it to a pre-registered falsifier or named control. Re-run, NOT a journal read; see FINDING 3. What a RED does NOT settle is whether the SCOPE is right - that is the spot-check |
| `tests/cameron/test_r6_matcher_red.py` | CAMERON round 6 it.0 -- the RED tests behind the Hungarian matcher (1.5). | NO JOURNAL ENTRY | live | KEEP | 14 tests, input drawn from the scale/ production path |
| `tests/cameron/test_r6_trained_red.py` | CAMERON Phase D -- the oldest open item: every probe number was random-init. | NO JOURNAL ENTRY | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/cameron/test_r6_twodof_red.py` | CAMERON round 6 -- the 2-dof lemma and its two degenerate loci (contract 1.3). | NO JOURNAL ENTRY | live | KEEP | 9 tests, input drawn from the scale/ production path |
| `tests/cameron/test_s2_ablation.py` | S2 — the selection ablation. Kills pre-registered in DONE.md BEFORE any arm ran. | NO JOURNAL ENTRY | live | KEEP | 9 tests, input drawn from the scale/ production path |
| `tests/cameron/test_schedule.py` | Is there a cheaper structure than a bespoke H-matrix schedule builder? | NO JOURNAL ENTRY | live | KEEP | MEASURED RED, presumption WITHDRAWN: this file has a demonstrated live rejection region - it is currently rejecting, so it is not passing by construction and cannot be vacuous in the strong sense. Its docstring binds it to a pre-registered falsifier or named control. Re-run, NOT a journal read; see FINDING 3. What a RED does NOT settle is whether the SCOPE is right - that is the spot-check |
| `tests/cameron/test_schedule_ablation.py` | What is the cheapest schedule that beats the 0D-salience control? | NO JOURNAL ENTRY | live | KEEP | MEASURED RED, presumption WITHDRAWN: this file has a demonstrated live rejection region - it is currently rejecting, so it is not passing by construction and cannot be vacuous in the strong sense. Its docstring binds it to a pre-registered falsifier or named control. Re-run, NOT a journal read; see FINDING 3. What a RED does NOT settle is whether the SCOPE is right - that is the spot-check |
| `tests/cameron/test_the_signed_arm_is_signed_on_this_task.py` | THE PRE-REGISTERED KILL: frustration(A[P,P]) >= 0.30 on the M3 harness data. | 2/3 readings reproduce at abs=5e-7 | live | KEEP | 3 tests, input drawn from the scale/ production path |
| `tests/cameron/test_u1_rag_registration.py` | U1 -- THE RAG-MULTIHOP TWIN REGISTRATION (contract v10.1, U-layer). | NO JOURNAL ENTRY | live | KEEP | 7 tests, input drawn from the scale/ production path |
| `tests/cameron/test_verdict_guards_partial_ladders.py` | RED-first bind for D-1, D-2 and H-1 in `PREREGISTRATION_HOLE_AUDIT.md`. | NO JOURNAL ENTRY | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/chase/axes.py` | The trained half of `test_scale_axes.py`, run once, recorded to JSONL. | NO JOURNAL ENTRY | live | KEEP | helper imported by 1 python module(s) |
| `tests/chase/conftest.py` | Shared fixtures for tests/chase. | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/chase/k22.py` | Forward attention over topology-derived causal CSR block schedules. | NO JOURNAL ENTRY | live | KEEP | helper imported by 4 python module(s) |
| `tests/chase/structural_zero.py` | Structural-zero guard: one check for the class that outnumbers every other | NO JOURNAL ENTRY | live | KEEP | helper imported by 1 python module(s) |
| `tests/chase/test_capability_table.py` | CHASE round 7 it.9 - the capability table v0, and the rule that makes it honest. | results/m3_quintuple_v2.jsonl (151 rec); results/m3_quintuple_v2_cuda.jsonl (23 rec) | live | KEEP | 15 tests, input drawn from the scale/ production path |
| `tests/chase/test_ceq_hub_package.py` | The Hub package: can somebody actually load and run this checkpoint. | NO JOURNAL ENTRY | live | KEEP | MEASURED RED, presumption WITHDRAWN: this file has a demonstrated live rejection region - it is currently rejecting, so it is not passing by construction and cannot be vacuous in the strong sense. Its docstring binds it to a pre-registered falsifier or named control. Re-run, NOT a journal read; see FINDING 3. What a RED does NOT settle is whether the SCOPE is right - that is the spot-check |
| `tests/chase/test_checklist_kills_are_evaluable.py` | CHECKLIST kill clauses that cannot fire. RED ON PURPOSE. | NO JOURNAL ENTRY | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/chase/test_colab_chain.py` | The Colab -> Hugging Face chain, end to end, without spending a GPU-hour. | NO JOURNAL ENTRY | live | KEEP | 11 tests, input drawn from the scale/ production path |
| `tests/chase/test_deq_divergence.py` | THEORY.md §8 risk 1 and the DEQ family's documented weakness, bound to a run. | NO JOURNAL ENTRY | live | KEEP | AMENDED r10-it2 (was vacuous/ATTIC): named 6 time(s) on `tests/chase/conftest.py`'s KNOWN_RED ledger, which resolves the key against collected node ids and applies xfail(strict=True) at EVERY collection. That is a live reader holding a live reference, and each entry carries a measured finding about the shipped deliverable. The iteration-1 cell read "ran ALL-GREEN"; a file whose every test is a strict xfail reports exit 0 and zero failures, so ALL-GREEN was read off an exit status that cannot tell a pass from a recorded finding. RED-BY-DESIGN, not vacuous. |
| `tests/chase/test_eprocess.py` | Binds on the anytime-valid e-process for the M3 settled-vs-twin contrast. | results/m3_quintuple.jsonl (19 rec) | live | KEEP | 31 tests, input drawn from the scale/ production path |
| `tests/chase/test_eprocess_perdraw.py` | X18 must-fire: the per-draw Ville process, calibrated in BOTH directions. | results/m3_quintuple_v2.jsonl (151 rec); results/m3_quintuple_v2_cuda.jsonl (23 rec) | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/chase/test_floor_objective.py` | THEORY.md §5: caustic Theorem 1 as a TRAINING objective rather than a diagnostic. | NO JOURNAL ENTRY | live | KEEP | AMENDED r10-it2 (was vacuous/ATTIC): named 6 time(s) on `tests/chase/conftest.py`'s KNOWN_RED ledger, which resolves the key against collected node ids and applies xfail(strict=True) at EVERY collection. That is a live reader holding a live reference, and each entry carries a measured finding about the shipped deliverable. The iteration-1 cell read "ran ALL-GREEN"; a file whose every test is a strict xfail reports exit 0 and zero failures, so ALL-GREEN was read off an exit status that cannot tell a pass from a recorded finding. RED-BY-DESIGN, not vacuous. |
| `tests/chase/test_hf_shipping.py` | What breaks when the target deliverable -- a ~1B model or module on the Hugging Face | NO JOURNAL ENTRY | live | KEEP | AMENDED r10-it2 (was vacuous/ATTIC): named 5 time(s) on `tests/chase/conftest.py`'s KNOWN_RED ledger, which resolves the key against collected node ids and applies xfail(strict=True) at EVERY collection. That is a live reader holding a live reference, and each entry carries a measured finding about the shipped deliverable. The iteration-1 cell read "ran ALL-GREEN"; a file whose every test is a strict xfail reports exit 0 and zero failures, so ALL-GREEN was read off an exit status that cannot tell a pass from a recorded finding. RED-BY-DESIGN, not vacuous. |
| `tests/chase/test_hub_package_hardening.py` | Round 4: make `ceq/hf/` publishable, and make every claim in it survive a | NO JOURNAL ENTRY | live | KEEP | 26 tests, input drawn from the scale/ production path |
| `tests/chase/test_kernel_contracts.py` | tda-tdd kernel correctness contracts applied to the ACTUAL merged kernel. | NO JOURNAL ENTRY | live | KEEP | AMENDED r10-it2 (was vacuous/ATTIC): named 9 time(s) on `tests/chase/conftest.py`'s KNOWN_RED ledger, which resolves the key against collected node ids and applies xfail(strict=True) at EVERY collection. That is a live reader holding a live reference, and each entry carries a measured finding about the shipped deliverable. The iteration-1 cell read "ran ALL-GREEN"; a file whose every test is a strict xfail reports exit 0 and zero failures, so ALL-GREEN was read off an exit status that cannot tell a pass from a recorded finding. RED-BY-DESIGN, not vacuous. |
| `tests/chase/test_lean_refcount_binding.py` | Bind `lean/CEQ/Refcount.lean` -- the only live provenance claim -- to a test. | NO JOURNAL ENTRY | live | KEEP | 10 tests, input drawn from the scale/ production path |
| `tests/chase/test_m2_instrument_binds.py` | Three binds M2 has to survive before it can be marked GREEN. ALL RED. | results/m2.jsonl (40 rec) | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/chase/test_m2_not_in_P_is_structural_zero.py` | Is the M2 `pivot_signed` / `not_in_P` arm a MEASUREMENT or an IDENTITY? | NO JOURNAL ENTRY | live | KEEP | 1 tests, input drawn from the scale/ production path |
| `tests/chase/test_m2_verdict_nan.py` | NaN control slope vs the M2 verdict. THIS FILE IS RED ON PURPOSE. | NO JOURNAL ENTRY | live | KEEP | 4 tests, input drawn from the scale/ production path |
| `tests/chase/test_m3_capability_harness.py` | CHASE round 3 - can the rebuilt M3 CAPABILITY harness print a wrong number? | NO JOURNAL ENTRY | live | KEEP | 11 tests, input drawn from the scale/ production path |
| `tests/chase/test_m3_ladder_task.py` | RED-first bind for the two things `scale/m3_quintuple.py` was missing. | results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | 14 tests, input drawn from the scale/ production path |
| `tests/chase/test_m3_quintuple_gradient.py` | CHASE Phase C - is the gradient the money run trains through actually right? | 1/1 readings reproduce at abs=5e-7 | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/chase/test_m3_quintuple_single_seed_table.py` | RED-first bind: the per-cell summary table survives a ONE-seed run. | results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | 2 tests, input drawn from the scale/ production path |
| `tests/chase/test_m3_synthetic_settled.py` | CHASE round 6 it.0 - can the M3 harness detect a PLANTED settled-vs-twin gap? | NO JOURNAL ENTRY | live | KEEP | 13 tests, input drawn from the scale/ production path |
| `tests/chase/test_module_prose_is_bound.py` | Two numbers a module docstring states and no test has ever checked. | 2/11 readings reproduce at abs=5e-7 | live | KEEP | builds its own input, BUT 2/11 of its readings reproduce from the journal at abs=5e-7 - its scope is anchored to the production path, so the L-SCOPE presumption does not fire |
| `tests/chase/test_multizoom_cost.py` | C5 and C6, measured. Utilisation, wall clock, and memory against a tuned baseline. | NO JOURNAL ENTRY | vacuous | ATTIC | PRESUMED vacuous-by-scope (L-SCOPE): builds its own tensor/graph instead of drawing from the production batch path, and 0 of its 0 6-significant-digit readings reproduce from any results/*.jsonl at abs=5e-7. NOT RUN in the iteration-1 sweep (does not collect as a set). PRESUMPTION, NOT VERDICT - iteration-2 spot-check candidate; see KILL 2 |
| `tests/chase/test_multizoom_kernel.py` | tda-tdd kernel correctness contracts for the multi-zoom attention kernel. | NO JOURNAL ENTRY | vacuous | ATTIC | PRESUMED vacuous-by-scope (L-SCOPE): builds its own tensor/graph instead of drawing from the production batch path, and 0 of its 0 6-significant-digit readings reproduce from any results/*.jsonl at abs=5e-7. NOT RUN in the iteration-1 sweep (does not collect as a set). PRESUMPTION, NOT VERDICT - iteration-2 spot-check candidate; see KILL 2 |
| `tests/chase/test_multizoom_r5.py` | R5's falsifiers. These decide whether R5 lives or is deleted. | NO JOURNAL ENTRY | vacuous | ATTIC | PRESUMED vacuous-by-scope (L-SCOPE): builds its own tensor/graph instead of drawing from the production batch path, and 0 of its 0 6-significant-digit readings reproduce from any results/*.jsonl at abs=5e-7. NOT RUN in the iteration-1 sweep (does not collect as a set). PRESUMPTION, NOT VERDICT - iteration-2 spot-check candidate; see KILL 2 |
| `tests/chase/test_pivot_exclusion_lift.py` | The pivot-exclusion lift: does `av` actually contain softmax's own row? | NO JOURNAL ENTRY | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/chase/test_resume_checkpoint.py` | PROOF that `train()` is resumable, cpu-only and tiny. | NO JOURNAL ENTRY | live | KEEP | 3 tests, input drawn from the scale/ production path |
| `tests/chase/test_rollback_flex_attention.py` | The conservative option for §8 risk 6, tested rather than asserted. | NO JOURNAL ENTRY | live | KEEP | AMENDED r10-it2 (was vacuous/ATTIC): named 1 time(s) on `tests/chase/conftest.py`'s KNOWN_RED ledger, which resolves the key against collected node ids and applies xfail(strict=True) at EVERY collection. That is a live reader holding a live reference, and each entry carries a measured finding about the shipped deliverable. The iteration-1 cell read "ran ALL-GREEN"; a file whose every test is a strict xfail reports exit 0 and zero failures, so ALL-GREEN was read off an exit status that cannot tell a pass from a recorded finding. RED-BY-DESIGN, not vacuous. |
| `tests/chase/test_scale_axes.py` | Does the 1.0334 parity ratio survive to 300M, or is it a 3.3M artifact? | NO JOURNAL ENTRY | live | KEEP | MEASURED RED, presumption WITHDRAWN: this file has a demonstrated live rejection region - it is currently rejecting, so it is not passing by construction and cannot be vacuous in the strong sense. Its docstring binds it to a pre-registered falsifier or named control. Re-run, NOT a journal read; see FINDING 3. What a RED does NOT settle is whether the SCOPE is right - that is the spot-check |
| `tests/chase/test_scale_hazards.py` | Failure modes that do not appear at 3.3M and do appear at 300M. | 1/2 readings reproduce at abs=5e-7 | live | KEEP | builds its own input, BUT 1/2 of its readings reproduce from the journal at abs=5e-7 - its scope is anchored to the production path, so the L-SCOPE presumption does not fire |
| `tests/chase/test_scale_sizing.py` | Does a 0.5B model with this attention fit a Colab GPU, and what does it cost? | NO JOURNAL ENTRY | live | KEEP | 21 tests, input drawn from the scale/ production path |
| `tests/chase/test_schedule_rebuild.py` | THEORY.md's single new component, attacked directly. | NO JOURNAL ENTRY | live | KEEP | AMENDED r10-it2 (was vacuous/ATTIC): named 5 time(s) on `tests/chase/conftest.py`'s KNOWN_RED ledger, which resolves the key against collected node ids and applies xfail(strict=True) at EVERY collection. That is a live reader holding a live reference, and each entry carries a measured finding about the shipped deliverable. The iteration-1 cell read "ran ALL-GREEN"; a file whose every test is a strict xfail reports exit 0 and zero failures, so ALL-GREEN was read off an exit status that cannot tell a pass from a recorded finding. RED-BY-DESIGN, not vacuous. |
| `tests/chase/test_signed_operator_trainability.py` | What breaks in TRAINING that did not break in inference. | NO JOURNAL ENTRY | live | KEEP | 15 tests, input drawn from the scale/ production path |
| `tests/chase/test_sprt_k1.py` | CHASE round 6 it.0 - the SPRT that is meant to close K1's sign-flip clause. | NO JOURNAL ENTRY | live | KEEP | 18 tests, input drawn from the scale/ production path |
| `tests/chase/test_stochastic_P.py` | THEORY.md §1 constraint 2 ("Make P a transition kernel: row-stochastic, non-negative") | NO JOURNAL ENTRY | live | KEEP | AMENDED r10-it2 (was vacuous/ATTIC): named 4 time(s) on `tests/chase/conftest.py`'s KNOWN_RED ledger, which resolves the key against collected node ids and applies xfail(strict=True) at EVERY collection. That is a live reader holding a live reference, and each entry carries a measured finding about the shipped deliverable. The iteration-1 cell read "ran ALL-GREEN"; a file whose every test is a strict xfail reports exit 0 and zero failures, so ALL-GREEN was read off an exit status that cannot tell a pass from a recorded finding. RED-BY-DESIGN, not vacuous. |
| `tests/chase/test_structural_zero_guard.py` | Demonstrates `structural_zero.assert_perturbation_moves_output` firing RED | NO JOURNAL ENTRY | live | KEEP | 4 tests, input drawn from the scale/ production path |
| `tests/chase/theory_ref.py` | Minimal faithful transcription of THEORY.md, so that tests have something to be RED against. | NO JOURNAL ENTRY | live | KEEP | helper imported by 4 python module(s) |
| `tests/deimos/test_deimos_r9_iteration1.py` | DEIMOS / MORIARTY's moon, R9 iteration 1 -- the three surfaces Mars left | NO JOURNAL ENTRY | live | KEEP | 8 tests, input drawn from the scale/ production path |
| `tests/foreman/_ceq.py` | Standalone kernel for the FOREMAN R1/R2 differential. | NO JOURNAL ENTRY | live | KEEP | helper imported by 3 python module(s) |
| `tests/foreman/_device.py` | Device parametrization for every FOREMAN test. | NO JOURNAL ENTRY | live | KEEP | helper imported by 15 python module(s) |
| `tests/foreman/_lib.py` | Shared bench for the FOREMAN differential on THEORY.md. | NO JOURNAL ENTRY | live | KEEP | helper imported by 6 python module(s) |
| `tests/foreman/conftest.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/foreman/test_absolute_floor_is_an_arm_filter.py` | An absolute threshold on a scale-free question is an arm-dependent filter. | NO JOURNAL ENTRY | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/foreman/test_capability_and_theorem_share_no_object.py` | The COGS capability number and the content-conditional-sign theorem have | 1/1 readings reproduce at abs=5e-7 | live | KEEP | 7 tests, input drawn from the scale/ production path |
| `tests/foreman/test_consequence_fidelity.py` | RED-first bind for CONSEQUENCE FIDELITY (`LOOP_PROMPT.md` 1.7c). | results/foreman_consequence.jsonl (52 rec) | live | KEEP | 7 tests, input drawn from the scale/ production path |
| `tests/foreman/test_deq_redundancy.py` | THEORY.md sec 8 risk 1: 'Nested fixed points. (I - gamma P)^-1 is itself a Bellman | NO JOURNAL ENTRY | live | KEEP | REFUTATION INSTRUMENT: defines test_claim_* functions, so its RED is its deliverable, not a defect. It constructs its own input by necessity - the mechanism it races is a competitor's, reimplemented from the defining equations, and a competitor has no production batch path here. P1' is NOT applied; see FINDING 2 |
| `tests/foreman/test_gromov_delta.py` | N2: the Gromov delta-hyperbolicity instrument, contract v10.1 section N2. | NO JOURNAL ENTRY | live | KEEP | 9 tests, input drawn from the scale/ production path |
| `tests/foreman/test_journal_thread_binding.py` | The M2 journal is REPLAYABLE. The unit key just does not name the thread count. | results/m2.jsonl (40 rec); results/s2.jsonl (13 rec) | live | KEEP | 4 tests, input drawn from the scale/ production path |
| `tests/foreman/test_l1_normalizer_obstruction.py` | Q2 -- is the pure operator trainable, and if not, what is the structural | NO JOURNAL ENTRY | live | KEEP | REFUTATION INSTRUMENT: defines test_claim_* functions, so its RED is its deliverable, not a defect. It constructs its own input by necessity - the mechanism it races is a competitor's, reimplemented from the defining equations, and a competitor has no production batch path here. P1' is NOT applied; see FINDING 2 |
| `tests/foreman/test_lorenz_occupancy.py` | THEORY.md sec 1, the sentence the author flags as load-bearing: | NO JOURNAL ENTRY | live | KEEP | REFUTATION INSTRUMENT: defines test_claim_* functions, so its RED is its deliverable, not a defect. It constructs its own input by necessity - the mechanism it races is a competitor's, reimplemented from the defining equations, and a competitor has no production batch path here. P1' is NOT applied; see FINDING 2 |
| `tests/foreman/test_lowpass_coupling.py` | THEORY.md lists these as two separate items, in two separate sections: | NO JOURNAL ENTRY | live | KEEP | REFUTATION INSTRUMENT: defines test_claim_* functions, so its RED is its deliverable, not a defect. It constructs its own input by necessity - the mechanism it races is a competitor's, reimplemented from the defining equations, and a competitor has no production batch path here. P1' is NOT applied; see FINDING 2 |
| `tests/foreman/test_m2_mechanism_story.py` | Pre-registered claims from scale/pivot_probe.py's module docstring, tested | results/m2.jsonl (40 rec) | live | KEEP | 4 tests, input drawn from the scale/ production path |
| `tests/foreman/test_m2prime_routes_are_independent.py` | M2' claims FOUR routes, any ONE sufficing. This file asserts that premise. | NO JOURNAL ENTRY | live | KEEP | 7 tests, input drawn from the scale/ production path |
| `tests/foreman/test_methods_mixture_identity.py` | Binds the one algebraically-falsifiable claim in `METHODS.md` §2. | NO JOURNAL ENTRY | live | KEEP | 4 tests, input drawn from the scale/ production path |
| `tests/foreman/test_operator_is_a_triangular_solve.py` | Q1 -- is `out = v + Av + A^2 v + ... + A^K v`, A strictly lower triangular, | NO JOURNAL ENTRY | live | KEEP | REFUTATION INSTRUMENT: defines test_claim_* functions, so its RED is its deliverable, not a defect. It constructs its own input by necessity - the mechanism it races is a competitor's, reimplemented from the defining equations, and a competitor has no production batch path here. P1' is NOT applied; see FINDING 2 |
| `tests/foreman/test_oracle_separation_binding.py` | `CEQ.OracleSeparation` must be compiled, and its hypotheses must hold of the | NO JOURNAL ENTRY | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/foreman/test_paraformer_ratio_across_context.py` | FOREMAN round 4 -- bind the ParaFormer separation across context length. | 2/3 readings reproduce at abs=5e-7 | live | KEEP | REFUTATION INSTRUMENT: defines test_claim_* functions, so its RED is its deliverable, not a defect. It constructs its own input by necessity - the mechanism it races is a competitor's, reimplemented from the defining equations, and a competitor has no production batch path here. P1' is NOT applied; see FINDING 2 |
| `tests/foreman/test_pivot_plateau_scales_with_k.py` | Pre-registered mechanism prediction: the pivot plateau height is set by the | NO JOURNAL ENTRY | live | KEEP | 2 tests, input drawn from the scale/ production path |
| `tests/foreman/test_pivot_selector_is_content_bearing.py` | Pre-registered claims from the M2 docstring (scale/pivot_probe.py), tested | NO JOURNAL ENTRY | live | KEEP | 3 tests, input drawn from the scale/ production path |
| `tests/foreman/test_ppnp_parity.py` | THEORY.md sec 0 claims the resolvent stage is 'exists, mislabelled' -- an asset | NO JOURNAL ENTRY | live | KEEP | REFUTATION INSTRUMENT: defines test_claim_* functions, so its RED is its deliverable, not a defect. It constructs its own input by necessity - the mechanism it races is a competitor's, reimplemented from the defining equations, and a competitor has no production batch path here. P1' is NOT applied; see FINDING 2 |
| `tests/foreman/test_r1_settling.py` | R1 -- "settle, don't glance". | NO JOURNAL ENTRY | live | KEEP | MEASURED RED, presumption WITHDRAWN: this file has a demonstrated live rejection region - it is currently rejecting, so it is not passing by construction and cannot be vacuous in the strong sense. Its docstring binds it to a pre-registered falsifier or named control. Re-run, NOT a journal read; see FINDING 3. What a RED does NOT settle is whether the SCOPE is right - that is the spot-check |
| `tests/foreman/test_r2_maxplus_reduction.py` | R2 -- "weigh by consequence, not similarity". | NO JOURNAL ENTRY | live | KEEP | MEASURED RED, presumption WITHDRAWN: this file has a demonstrated live rejection region - it is currently rejecting, so it is not passing by construction and cannot be vacuous in the strong sense. Its docstring binds it to a pre-registered falsifier or named control. Re-run, NOT a journal read; see FINDING 3. What a RED does NOT settle is whether the SCOPE is right - that is the spot-check |
| `tests/foreman/test_r2_signed_consequence_fit.py` | R2's falsifier, with a learned weighting rather than a sampled Jacobian. | NO JOURNAL ENTRY | live | KEEP | 4 tests, input drawn from the scale/ production path |
| `tests/foreman/test_resolvent_certificate.py` | THEORY.md sec 1: 'With P row-stochastic, rho(gamma P) = gamma < 1 holds *by | NO JOURNAL ENTRY | live | KEEP | REFUTATION INSTRUMENT: defines test_claim_* functions, so its RED is its deliverable, not a defect. It constructs its own input by necessity - the mechanism it races is a competitor's, reimplemented from the defining equations, and a competitor has no production batch path here. P1' is NOT applied; see FINDING 2 |
| `tests/foreman/test_sgate_scaling_defect.py` | FOREMAN round 3 -- is 1.0334 at 3.3M a property of the operator or of the box? | 1/5 readings reproduce at abs=5e-7 | live | KEEP | REFUTATION INSTRUMENT: defines test_claim_* functions, so its RED is its deliverable, not a defect. It constructs its own input by necessity - the mechanism it races is a competitor's, reimplemented from the defining equations, and a competitor has no production batch path here. P1' is NOT applied; see FINDING 2 |
| `tests/foreman/test_sheaf_kernel_parity.py` | THEORY.md sec 8 risk 2: 'ker Delta_F may be trivial. For a connected sheaf with | NO JOURNAL ENTRY | live | KEEP | REFUTATION INSTRUMENT: defines test_claim_* functions, so its RED is its deliverable, not a defect. It constructs its own input by necessity - the mechanism it races is a competitor's, reimplemented from the defining equations, and a competitor has no production batch path here. P1' is NOT applied; see FINDING 2 |
| `tests/foreman/test_sign_floor.py` | The sign floor: a certified threshold below which `sgate` cannot be signed. | results/foreman_signfloor.jsonl (30 rec) | live | KEEP | 9 tests, input drawn from the scale/ production path |
| `tests/foreman/test_signedness_is_not_new.py` | Q3 -- is SIGNEDNESS alone enough to be new? | NO JOURNAL ENTRY | live | KEEP | REFUTATION INSTRUMENT: defines test_claim_* functions, so its RED is its deliverable, not a defect. It constructs its own input by necessity - the mechanism it races is a competitor's, reimplemented from the defining equations, and a competitor has no production batch path here. P1' is NOT applied; see FINDING 2 |
| `tests/foreman/test_topology_washout.py` | Is the topology load-bearing, or is it vocabulary wrapped around linear algebra? | NO JOURNAL ENTRY | live | KEEP | REFUTATION INSTRUMENT: defines test_claim_* functions, so its RED is its deliverable, not a defect. It constructs its own input by necessity - the mechanism it races is a competitor's, reimplemented from the defining equations, and a competitor has no production batch path here. P1' is NOT applied; see FINDING 2 |
| `tests/foreman/test_window_width_is_the_missing_control.py` | S1's missing control, pinned as a test so it cannot go missing again. | NO JOURNAL ENTRY | vacuous | ATTIC | PRESUMED vacuous-by-scope (L-SCOPE): builds its own tensor/graph instead of drawing from the production batch path, and 0 of its 0 6-significant-digit readings reproduce from any results/*.jsonl at abs=5e-7. ran ALL-GREEN in the iteration-1 sweep, which is exactly what a vacuous control does and is therefore not evidence either way. PRESUMPTION, NOT VERDICT - iteration-2 spot-check candidate; see KILL 2 |
| `tests/gpu/__init__.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/gpu/test_cuda_parity.py` | CUDA parity lane for the m3 arms: softmax, twin, settled (and their controls). | results/m3_quintuple_v2.jsonl (151 rec); results/m3_quintuple_v2_cuda.jsonl (23 rec) | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/jupiter/__init__.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/jupiter/test_coherence_and_rip.py` | The second path for M1/M5's coherence arithmetic and M2's RIP line. | NO JOURNAL ENTRY | live | KEEP | 7 tests, input drawn from the scale/ production path |
| `tests/jupiter/test_kirchhoff_agreement.py` | The instrument law: two independent oracles for the same Dirichlet solution. | NO JOURNAL ENTRY | live | KEEP | 10 tests, input drawn from the scale/ production path |
| `tests/jupiter/test_page_lattice.py` | The granularity of the trend clause, against Venus's floor for the size clause. | results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | 8 tests, input drawn from the scale/ production path |
| `tests/jupiter/test_page_trend.py` | The trend verdict's own error rates, measured on drawn tables. | results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | 9 tests, input drawn from the scale/ production path |
| `tests/jupiter/test_pivot_selection_theory.py` | The two derivations `results/r9_maths_survey.md` rests on, as runnable checks. | NO JOURNAL ENTRY | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/jupiter/test_struck_coverage_scans.py` | `scale/chase_struck_coverage.py` must actually walk files, and be shown to. | NO JOURNAL ENTRY | live | KEEP | 8 tests, input drawn from the scale/ production path |
| `tests/loop/test_arms_distinct.py` | ARMS-DISTINCT: no arm may report another arm's number under its own name. | 2/2 readings reproduce at abs=5e-7 | live | KEEP | 8 tests, input drawn from the scale/ production path |
| `tests/loop/test_bootstrap_is_position_independent.py` | A bootstrap helper must not depend on where it sits in a shared stream. | NO JOURNAL ENTRY | live | KEEP | 4 tests, input drawn from the scale/ production path |
| `tests/loop/test_dispatch_count.py` | The cost gate's dispatch claim, made testable. | NO JOURNAL ENTRY | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/loop/test_fused_settle.py` | S6: the settling loop's per-step dispatch cost, and what fusing may not change. | NO JOURNAL ENTRY | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/loop/test_hilbert_metric.py` | The Hilbert projective metric, bound to its defining properties. | NO JOURNAL ENTRY | live | KEEP | 25 tests, input drawn from the scale/ production path |
| `tests/loop/test_journal_scan.py` | A journal scan that cannot silently fail to find what it is searching for. | NO JOURNAL ENTRY | live | KEEP | 7 tests, input drawn from the scale/ production path |
| `tests/loop/test_l3_leakage.py` | Contract 1.3's leakage gate, computed rather than swept. | NO JOURNAL ENTRY | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/loop/test_m3_harness_operator_is_shipped.py` | The CAPABILITY harness must measure an operator the module ships. | NO JOURNAL ENTRY | live | KEEP | 4 tests, input drawn from the scale/ production path |
| `tests/loop/test_m4_eviction_is_calibrated.py` | M4's kill, calibrated at both ends — the must-fire arm it never had. | 2/2 readings reproduce at abs=5e-7 | live | KEEP | builds its own input, BUT 2/2 of its readings reproduce from the journal at abs=5e-7 - its scope is anchored to the production path, so the L-SCOPE presumption does not fire |
| `tests/loop/test_measured_operator_is_shipped.py` | Every operator a probe MEASURES must be one the module SHIPS, or be declared. | NO JOURNAL ENTRY | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/loop/test_merkle_append.py` | A sealed journal that legitimately grows must be distinguishable from one edited. | NO JOURNAL ENTRY | live | KEEP | 9 tests, input drawn from the scale/ production path |
| `tests/loop/test_merkle_journal.py` | The Merkle journal, and the tamper test that makes it worth having. | NO JOURNAL ENTRY | live | KEEP | 11 tests, input drawn from the scale/ production path |
| `tests/loop/test_monge_oracle.py` | Rectangular Monge oracle, bound against an independent solver. | NO JOURNAL ENTRY | live | KEEP | 9 tests, input drawn from the scale/ production path |
| `tests/loop/test_no_struck_constant_ships.py` | No number this project has struck may be asserted by shipped code or a lead document. | 1/9 readings reproduce at abs=5e-7 | live | KEEP | pre-seeded KEEP: THE STRUCK REGISTRY. It lives HERE, in the module-level `STRUCK: dict[float, str]` at :47 - there is no STRUCK.md at root. inspector.py:459 asserts it is non-empty |
| `tests/loop/test_pivot_arms_distinct.py` | ARMS-DISTINCT for `scale/pivot_probe.py`. G3, reopened in a new file. | NO JOURNAL ENTRY | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/loop/test_settle.py` | The settling driver, bound to the convergence law it claims to verify. | NO JOURNAL ENTRY | live | KEEP | 10 tests, input drawn from the scale/ production path |
| `tests/loop/test_theorem_hypotheses_hold_at_shipped_settings.py` | Every Lean theorem cited as certifying the module must hold AT ITS SETTINGS. | NO JOURNAL ENTRY | vacuous | ATTIC | PRESUMED vacuous-by-scope (L-SCOPE): builds its own tensor/graph instead of drawing from the production batch path, and 0 of its 9 6-significant-digit readings reproduce from any results/*.jsonl at abs=5e-7. ran ALL-GREEN in the iteration-1 sweep, which is exactly what a vacuous control does and is therefore not evidence either way. PRESUMPTION, NOT VERDICT - iteration-2 spot-check candidate; see KILL 2 |
| `tests/loop/test_two_dof_lemma.py` | The 2-dof lemma: why the four-point probe cannot collapse the way round 5 did. | 1/1 readings reproduce at abs=5e-7 | live | KEEP | builds its own input, BUT 1/1 of its readings reproduce from the journal at abs=5e-7 - its scope is anchored to the production path, so the L-SCOPE presumption does not fire |
| `tests/mars/test_mars_argmax_ste.py` | MARS / MORIARTY, R9 iteration 2 -- the bind on the straight-through cell. | results/m3_quintuple_v2.jsonl (151 rec); results/m3_quintuple_v2_cuda.jsonl (23 rec) | live | KEEP | 8 tests, input drawn from the scale/ production path |
| `tests/mars/test_mars_control_entry_point.py` | MARS / MORIARTY, R9 iteration 3 -- the mechanized catch for the new class. | NO JOURNAL ENTRY | live | KEEP | 4 tests, input drawn from the scale/ production path |
| `tests/mars/test_mars_green_attacks.py` | MARS / MORIARTY, R9 iteration 3 -- attacks filed on the standing GREENs. | results/arm_s.jsonl (82 rec); results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | 3 tests, input drawn from the scale/ production path |
| `tests/mars/test_mars_green_attacks_it4.py` | MARS / MORIARTY, R9 iteration 4 -- the three GREENs nobody had attacked. | 6/17 readings reproduce at abs=5e-7 | live | KEEP | 4 tests, input drawn from the scale/ production path |
| `tests/mars/test_mars_r9_iteration1.py` | MARS / MORIARTY, R9 iteration 1 -- the standing adversary's battery. | results/m3_quintuple_v2.jsonl (151 rec); results/m3_quintuple_v2_cuda.jsonl (23 rec) | live | KEEP | 8 tests, input drawn from the scale/ production path |
| `tests/mercury/test_r9_eprocess_overflow.py` | Regression battery for the e-process overflow and its untested branch. | NO JOURNAL ENTRY | live | KEEP | 14 tests, input drawn from the scale/ production path |
| `tests/mercury/test_r9_estimator_named.py` | Every interval the card renders must name the estimator that produced it. | NO JOURNAL ENTRY | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/mercury/test_r9_exact_interval.py` | A 5-seed bootstrap interval lands on a lattice; report where, exactly. | 10/14 readings reproduce at abs=5e-7 | live | KEEP | 7 tests, input drawn from the scale/ production path |
| `tests/mercury/test_r9_limits_e.py` | Limits paragraph (e) must describe the estimator, not audit other files. | results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/mercury/test_r9_mechanical_fixes.py` | Regression battery for the four mechanical defects fixed in R9 iteration 1. | NO JOURNAL ENTRY | live | KEEP | 14 tests, input drawn from the scale/ production path |
| `tests/mercury/test_r9_seed_agreement.py` | Every 5-seed interval must carry the seed-agreement count that produced it. | NO JOURNAL ENTRY | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/mercury/test_r9_table_cut.py` | Every count the capability card states must be counted, not stored. | results/m3_quintuple_v2.jsonl (151 rec); results/m3_quintuple_v2_cuda.jsonl (23 rec) | live | KEEP | 13 tests, input drawn from the scale/ production path |
| `tests/mercury/test_r9_verdict_names.py` | A verdict must name the cells it compared, not always `settled` and `twin`. | NO JOURNAL ENTRY | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/neptune/conftest.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/neptune/test_capability_table_truth.py` | The capability-table generator must not assert what its own artifact denies. | NO JOURNAL ENTRY | live | KEEP | 7 tests, input drawn from the scale/ production path |
| `tests/neptune/test_per_row_arm.py` | The per-row cells: the bind that must hold, and the RED that must move bits. | NO JOURNAL ENTRY | live | KEEP | 12 tests, input drawn from the scale/ production path |
| `tests/w10/conftest.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/w10/test_w10_from_scratch.py` | W10 -- the signed operator AS the attention, trained from scratch. | NO JOURNAL ENTRY | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/w11/conftest.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/w11/test_w11_claims_resolve.py` | W11 -- every claim in the README must name a test that exists. | NO JOURNAL ENTRY | live | KEEP | 11 tests, input drawn from the scale/ production path |
| `tests/w13/conftest.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/w13/test_w13_h1_selectivity.py` | H1 -- restore selectivity without losing the sign. | NO JOURNAL ENTRY | live | KEEP | 4 tests, input drawn from the scale/ production path |
| `tests/w14/conftest.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/w14/test_w14_identity_double_count.py` | H7 -- the identity term is counted twice, and only in the signed arms. | NO JOURNAL ENTRY | live | KEEP | 4 tests, input drawn from the scale/ production path |
| `tests/w15/conftest.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/w15/test_w15_rho_is_the_self_other_ratio.py` | H8 -- rho is the self/other weight ratio and it has never been tuned. | NO JOURNAL ENTRY | live | KEEP | 3 tests, input drawn from the scale/ production path |
| `tests/w2/conftest.py` | Shared fixtures for W2. | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/w2/test_w2_nonnormal.py` | W2 -- the non-normal operator. Replaces the deleted R3. | NO JOURNAL ENTRY | live | KEEP | 8 tests, input drawn from the scale/ production path |
| `tests/w3/conftest.py` | Shared fixtures for W2. | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/w3/test_w3_eviction.py` | W3 -- eviction as its own requirement. | 1/3 readings reproduce at abs=5e-7 | live | KEEP | 5 tests, input drawn from the scale/ production path |
| `tests/w3b/test_w3b_lean_nilpotent.py` | W3b -- machine-check the nilpotent resolvent. | NO JOURNAL ENTRY | live | KEEP | 6 tests, input drawn from the scale/ production path |
| `tests/w4/conftest.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/w4/test_w4_intervention.py` | W4 -- intervention generalization. The requirement the module exists for. | NO JOURNAL ENTRY | live | KEEP | 13 tests, input drawn from the scale/ production path |
| `tests/w6/conftest.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/w6/test_w6_attention.py` | W6.2a -- the CPU reference forward pass. | NO JOURNAL ENTRY | vacuous | ATTIC | PRESUMED vacuous-by-scope (L-SCOPE): builds its own tensor/graph instead of drawing from the production batch path, and 0 of its 0 6-significant-digit readings reproduce from any results/*.jsonl at abs=5e-7. ran ALL-GREEN in the iteration-1 sweep, which is exactly what a vacuous control does and is therefore not evidence either way. PRESUMPTION, NOT VERDICT - iteration-2 spot-check candidate; see KILL 2 |
| `tests/w7/conftest.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/w7/test_w7_nash.py` | W7 -- Nash-equilibrium attention over a set system. | NO JOURNAL ENTRY | live | KEEP | 9 tests, input drawn from the scale/ production path |
| `tests/w8/conftest.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/w8/test_w8_real_model.py` | W8 -- the surviving parts, against a real checkpoint. | NO JOURNAL ENTRY | live | KEEP | 10 tests, input drawn from the scale/ production path |
| `tests/w9/conftest.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/w9/test_w9_hopcache.py` | W9 -- a scoped, topological hop cache, so alpha > 0 can decode. | 4/4 readings reproduce at abs=5e-7 | live | KEEP | builds its own input, BUT 4/4 of its readings reproduce from the journal at abs=5e-7 - its scope is anchored to the production path, so the L-SCOPE presumption does not fire |
| `tests/watson/conftest.py` | (no docstring) | NO JOURNAL ENTRY | live | KEEP | pytest machinery, collected implicitly by every sibling module in its directory |
| `tests/watson/test_vgpe_binds.py` | WATSON -- the K1 gate for VGPE. If these fail, nothing else in the round runs. | 6/13 readings reproduce at abs=5e-7 | live | KEEP | 11 tests, input drawn from the scale/ production path |
| `tests/wilson/accum_divergence.py` | Does gradient accumulation change the answer? Measure it; do not assert it. | NO JOURNAL ENTRY | orphan | ATTIC | helper: 0 test functions, 0 python importers, no journal; see KILL 3 |
| `tests/wilson/hop2_vec.py` | Batched pivot hop-2, without the Python loop over the batch. | NO JOURNAL ENTRY | live | KEEP | helper imported by 1 python module(s) |
| `tests/wilson/test_hop2_vec.py` | BITWISE bind: the vectorised batched hop-2 against the Python loop it replaces. | NO JOURNAL ENTRY | live | KEEP | 6 tests, input drawn from the scale/ production path |

### scale/*.py

Count: **99** &mdash; `git ls-files -- 'scale/*.py'`

| path | claims to test | last-run state FROM THE JOURNAL | class | KEEP/ATTIC | one-line reason |
|---|---|---|---|---|---|
| `scale/aggregator_matched_filler.py` | The control that decides whether the aggregator finding is a result or the selector. | results/agg_matched.jsonl (4 rec); results/aggregator_mech.jsonl (3 rec) | live | KEEP | producer of results/agg_matched.jsonl |
| `scale/aggregator_mechanism.py` | What the aggregator win actually is, derived from the identity rather than guessed. | results/aggregator_mech.jsonl (3 rec); results/b1_collapse.jsonl (6 rec) | live | KEEP | producer of results/aggregator_mech.jsonl |
| `scale/arm_a_k1.py` | ARM A -- K1 re-evaluated with the flip half on a SIGNED operator. | NO JOURNAL ENTRY | live | KEEP | imported by 2 python module(s) |
| `scale/arm_a_rebuild.py` | ARM A, REBUILT. The old theorem was about DIFFERENCES; the object is SUMS. | NO JOURNAL ENTRY | live | KEEP | imported by 1 python module(s) |
| `scale/arm_a_run.py` | ARM A driver — K1, K2, K3. Pure measurement. Bucketed, pinned, declared. | results/arm_a.jsonl (3 rec) | live | KEEP | producer of results/arm_a.jsonl |
| `scale/arm_s.py` | ARM S -- the settling attention arm, built to its birth gates. | results/arm_s.jsonl (82 rec) | live | KEEP | producer of results/arm_s.jsonl |
| `scale/arm_s_batched.py` | ARM S settling, BATCHED over examples, bound against `scale/arm_s.py`. | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 3 tracked file(s) (CHECKLIST.md, DONE.md), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/b1_collapse_test.py` | ARM B's B1: is "shadow mass INSTEAD OF salience" actually a different criterion? | results/b1_collapse.jsonl (6 rec) | live | KEEP | producer of results/b1_collapse.jsonl |
| `scale/b1_decomposition.py` | B1, settled by decomposition: Wilson and the coordinator measured DIFFERENT objects. | results/b1_collapse.jsonl (6 rec); results/b1_decomposition.jsonl (4 rec) | live | KEEP | producer of results/b1_decomposition.jsonl |
| `scale/bucket.py` | Bucketed, resumable measurement units. See ADR-001. | NO JOURNAL ENTRY | live | KEEP | imported by 11 python module(s) |
| `scale/cameron_aggregator_probe.py` | CAMERON, round 5 -- the AGGREGATOR probe. Measurement only, nothing built. | results/arm_a.jsonl (3 rec); results/cameron_aggregators.jsonl (8 rec) | live | KEEP | producer of results/arm_a.jsonl, results/cameron_aggregators.jsonl |
| `scale/capability_table.py` | Capability table v0 -- the negatives included, from artifacts that already exist. | results/m3_quintuple_v2.jsonl (151 rec); results/m3_quintuple_v2_cuda.jsonl (23 rec) | live | KEEP | producer of results/m3_quintuple_v2.jsonl |
| `scale/carpet_probe.py` | Hierarchical (self-similar) reachability with hops scaled to depth. | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 8 tracked file(s) (DONE.md, DONE_ARCHIVE_ROUND1.md), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/chase_k2_salience.py` | CHASE — is K2's filler twin measuring consequence, or measuring the selector? | results/arm_a.jsonl (3 rec) | live | KEEP | producer of results/arm_a.jsonl |
| `scale/chase_k3_ci.py` | CHASE — K3 with the error bar the criterion never had. | results/arm_a.jsonl (3 rec) | live | KEEP | producer of results/arm_a.jsonl |
| `scale/chase_slope_ci.py` | CHASE — the D_FR slope's interval, which the record reads a verdict out of without. | results/arm_a.jsonl (3 rec) | live | KEEP | producer of results/arm_a.jsonl |
| `scale/chase_struck_coverage.py` | CHASE — what the struck-constant absence check does NOT look at. | NO JOURNAL ENTRY | live | KEEP | imported by 1 python module(s) |
| `scale/coherence_floor.py` | How far apart k random unit vectors in R^d actually sit, against three closed | 2/11 readings reproduce at abs=5e-7 | live | KEEP | AMENDED r10-it2: the iteration-1 cell read "no python importer"; re-measured with a front-door grep there are 1 (tests/jupiter/test_coherence_and_rip.py). Disposition unchanged. no python importer, but 2/11 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/dfloor_probe.py` | Dr House's falsifier: a CONSEQUENCE-scored selector, not a resemblance one. | 2/2 readings reproduce at abs=5e-7 | live | KEEP | no python importer, but 2/2 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/dfloor_probe2.py` | DeltaFloor falsifier, rebuilt. The first version's slope was an artifact. | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 4 tracked file(s) (ARSENAL.md, CONTRACT.md), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/difference_set_arm.py` | ARM A — the difference-set schedule, with its birth gates. | NO JOURNAL ENTRY | live | KEEP | imported by 2 python module(s) |
| `scale/dispatch_count.py` | Count aten dispatches, so the cost gate stops resting on an inference. | NO JOURNAL ENTRY | live | KEEP | imported by 2 python module(s) |
| `scale/e4_harmonic.py` | The e3-harmonic ladder on the E4' Rips graphs, and the band it cannot reach. | 4/16 readings reproduce at abs=5e-7 | live | KEEP | AMENDED r10-it2: the iteration-1 cell read "no python importer"; re-measured with a front-door grep there are 6 (scale/e4_harmonic_reroute.py, scale/kirchhoff.py, tests/cameron/test_e4_harmonic.py and 3 more). Disposition unchanged. no python importer, but 4/16 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/e4_harmonic_reroute.py` | The RULE 5 reroute for X17, measured rather than proposed. | 1/8 readings reproduce at abs=5e-7 | live | KEEP | no python importer, but 1/8 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/e_ladder.py` | LADDER E — the settled-twin dose-response across the `t*` rungs. One curve. | results/m3_quintuple_v2.jsonl (151 rec); results/m3_quintuple_v2_cuda.jsonl (23 rec) | live | KEEP | producer of results/m3_quintuple_v2.jsonl |
| `scale/eprocess.py` | The anytime-valid e-process for the M3 settled-vs-twin contrast. | results/m3_quintuple.jsonl (19 rec); results/m3_quintuple_v2.jsonl (151 rec) +1 | live | KEEP | producer of results/m3_quintuple.jsonl, results/m3_quintuple_v2.jsonl |
| `scale/eprocess_perdraw.py` | X18 — the PER-DRAW Ville e-process. The unit is the evaluation draw. | results/m3_quintuple_v2.jsonl (151 rec); results/m3_quintuple_v2_cuda.jsonl (23 rec) | live | KEEP | imported by 1 python module(s) |
| `scale/equilibrium_probe.py` | X6 -- the equilibrium clause, tested against its own pre-registered kill. | results/equilibrium.jsonl (9 rec) | live | KEEP | producer of results/equilibrium.jsonl |
| `scale/etask_k5e.py` | K-5E: can an arm read an EQUILIBRIUM label at all, and where is the hop wall? | results/m3_quintuple_v2.jsonl (151 rec); results/m3_quintuple_v2_cuda.jsonl (23 rec) | live | KEEP | imported by 1 python module(s) |
| `scale/fgreen_matched.py` | The F-green matched re-run: is the one positive result the router or the selector? | 4/6 readings reproduce at abs=5e-7 | live | KEEP | no python importer, but 4/6 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/foreman_consequence.py` | CONSEQUENCE FIDELITY — the capability metric that replaces the Hankel frame. | results/foreman_consequence.jsonl (52 rec); results/m3_quintuple_v2.jsonl (151 rec) +1 | live | KEEP | producer of results/foreman_consequence.jsonl |
| `scale/foreman_curvature.py` | FOREMAN probe 2 -- does the SPHERE do any work, or only the square root? | results/arm_a.jsonl (3 rec); results/foreman_theta_tv.jsonl (27 rec) | live | KEEP | producer of results/foreman_theta_tv.jsonl |
| `scale/foreman_gram.py` | Round 6 iteration 22 -- binding Dr House's leap, or killing it. | results/gram.jsonl (18 rec) | live | KEEP | producer of results/gram.jsonl |
| `scale/foreman_hilbert.py` | Round 6, iteration 0, Foreman. The Hilbert metric: positivity, and the size | results/hilbert.jsonl (142 rec) | live | KEEP | producer of results/hilbert.jsonl |
| `scale/foreman_lambda2.py` | Engineer and MEASURE the relaxation dial of the E4' absorbing chain. | 10/33 readings reproduce at abs=5e-7 | live | KEEP | AMENDED r10-it2: the iteration-1 cell read "no python importer"; re-measured with a front-door grep there are 1 (tests/foreman/test_oracle_separation_binding.py). Disposition unchanged. no python importer, but 10/33 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/foreman_looped.py` | MOVE 2 -- iterate the REPRESENTATION, not the mixture weights. | results/foreman_looped.jsonl (4 rec) | live | KEEP | producer of results/foreman_looped.jsonl |
| `scale/foreman_quantisation.py` | FOREMAN probe 3 -- theta is not measured, it is QUANTISED by float32 arccos. | results/arm_a.jsonl (3 rec); results/foreman_theta_tv.jsonl (27 rec) | live | KEEP | producer of results/foreman_theta_tv.jsonl |
| `scale/foreman_signfloor.py` | The sign floor: where `sgate` stops being signed, as a certificate not a reading. | results/foreman_signfloor.jsonl (30 rec) | live | KEEP | producer of results/foreman_signfloor.jsonl |
| `scale/foreman_theta_tv.py` | FOREMAN probe -- is theta a reparametrisation of TV, and what is in D_FR? | results/arm_a.jsonl (3 rec); results/foreman_theta_tv.jsonl (27 rec) | live | KEEP | producer of results/arm_a.jsonl, results/foreman_theta_tv.jsonl |
| `scale/frustration_audit.py` | Zaslavsky frustration: is a signed arm SIGNED, or signed in name only? | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 2 tracked file(s) (DONE.md, tests/cameron/test_the_signed_arm_is_signed_on_this_task.py), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/g7_event_change.py` | G7 — does the arm change the EVENT, or only the STATISTIC? Both arms, pre-build. | 1/1 readings reproduce at abs=5e-7 | live | KEEP | no python importer, but 1/1 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/gate3_audit.py` | Is gate 3's off-schedule cell a MEASUREMENT, or is it zero by construction? | 1/5 readings reproduce at abs=5e-7 | live | KEEP | no python importer, but 1/5 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/hilbert.py` | The Hilbert projective metric and the Birkhoff contraction constants. | 2/4 readings reproduce at abs=5e-7 | live | KEEP | AMENDED r10-it2: the iteration-1 cell read "no python importer"; re-measured with a front-door grep there are 7 (scale/arm_s.py, scale/foreman_gram.py, scale/foreman_hilbert.py and 4 more). Disposition unchanged. no python importer, but 2/4 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/hyperbolic.py` | Gromov delta-hyperbolicity for corpus graphs. Contract v10.1 section N2. | NO JOURNAL ENTRY | live | KEEP | imported by 1 python module(s) |
| `scale/identity_manifest.py` | X-R1: the identity manifest -- a content hash of what was actually measured. | NO JOURNAL ENTRY | live | KEEP | pre-seeded KEEP (proven this branch): refusal fires on five config fields naming the moved field; published cell settled_k8_..._t1 reproduces 0.9783142763084641 |
| `scale/impact.py` | IMPACT — planted news→asset propagation, T-FAMILY 2. | 3/3 readings reproduce at abs=5e-7 | live | KEEP | pre-seeded ATTIC applies to the TASK, not this file. impact / impact_hetero are CORPUS CANDIDATES retired on three measured grounds (linear-probe train 4.93e-08 vs eval 1.478 across a severed split; trivially linear within a graph; 8192 MiB needed on an 8188 MiB card). THE REPAIRED CODE STAYS - no journal says the code is dead. See KILL 1 |
| `scale/journal_census.py` | Which thread counts reproduce which journalled units? Bucketed, resumable. | results/journal_census.jsonl (4 rec); results/m2.jsonl (40 rec) | live | KEEP | producer of results/journal_census.jsonl, results/m2.jsonl |
| `scale/journal_scan.py` | A journal scan that cannot silently fail to find what it is searching for. | NO JOURNAL ENTRY | live | KEEP | imported by 1 python module(s) |
| `scale/k3_concavity_control.py` | K3's MISSING CONTROL ARM: a concave reparametrisation of TV with no geometry. | results/k3_concavity.jsonl (6 rec) | live | KEEP | producer of results/k3_concavity.jsonl |
| `scale/kirchhoff.py` | A second exact oracle for the E4' harmonic measure, from the matrix-tree theorem. | 3/4 readings reproduce at abs=5e-7 | live | KEEP | pre-seeded KEEP (proven this branch): dual oracle, agreement 2.220446e-16 / 8.992806e-15 / 9.636736e-14 at 9/62/1202 nodes; planted off-by-one moves omega by 0.175-0.316, caught 6/6 |
| `scale/lastrow_bind.py` | THE PROPOSED OPTIMISATION AND ITS BITWISE BIND, IN ONE FILE. | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 2 tracked file(s) (ceq/bench.py, scale/maskcache_bind.py), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/lo_probe.py` | Littlewood-Offord decomposition of the -1.389 exponent, and the KPZ | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 2 tracked file(s) (DONE_ARCHIVE_ROUND1.md, PREREGISTRATION_HOLE_AUDIT.md), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/m2_units.py` | M2 as idempotent units in wall-clock buckets. ADR-001. | results/m2.jsonl (40 rec) via Journal(NAME="m2") -> scale/bucket.py:45 | live | KEEP | producer of results/m2.jsonl - path CONSTRUCTED via Journal(NAME="m2") -> scale/bucket.py:45, invisible to a literal scan |
| `scale/m3_capability.py` | M3 capability arms — trainable ARMS on the calibrated M3 instrument. | NO JOURNAL ENTRY | live | KEEP | imported by 25 python module(s) |
| `scale/m3_flops.py` | Analytic FLOP accounting for the five M3 cells. | results/arm_s.jsonl (82 rec) | live | KEEP | imported by 3 python module(s) |
| `scale/m3_quintuple.py` | M3, the deciding measurement: five cells, batched, bucketed. | results/m3_quintuple_v2.jsonl (151 rec); results/m3_quintuple_v2_cuda.jsonl (23 rec) via Journal(NAME="m3_quintuple_v2") -> scale/bucket.py:45 | live | KEEP | producer of results/m3_quintuple_v2.jsonl, results/m3_quintuple_v2_cuda.jsonl - path CONSTRUCTED via Journal(NAME="m3_quintuple_v2") -> scale/bucket.py:45, invisible to a literal scan |
| `scale/m3_synthetic_settled.py` | M3 harness dry-run on SYNTHETIC arms whose verdict is known in advance. | 3/8 readings reproduce at abs=5e-7 | live | KEEP | AMENDED r10-it2: the iteration-1 cell read "no python importer"; re-measured with a front-door grep there are 11 (scale/capability_table.py, scale/e_ladder.py, scale/m3_quintuple.py and 8 more). Disposition unchanged. no python importer, but 3/8 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/maskcache_applied_bind.py` | POST-APPLICATION bind for the `_causal_mask_pair` memoisation in ceq/bench.py. | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers, 0 results artifacts, 0 prose citations in any tracked file. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/maskcache_bind.py` | CANDIDATE 3, after candidates 1 and 2 both FAILED the bitwise bind. | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 1 tracked file(s) (scale/maskcache_applied_bind.py), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/matcher.py` | The key-norm matcher that every causal-vs-filler contrast in this project owes. | 5/10 readings reproduce at abs=5e-7 | live | KEEP | AMENDED r10-it2: the iteration-1 cell read "no python importer"; re-measured with a front-door grep there are 2 (scale/fgreen_matched.py, tests/cameron/test_r6_matcher_red.py). Disposition unchanged. no python importer, but 5/10 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/max_row_mechanism.py` | Cameron's mechanism claim for the aggregator win, bound to a RED test. | results/max_row.jsonl (6 rec) | live | KEEP | producer of results/max_row.jsonl |
| `scale/mech_attack.py` | Mechanistic attack probe on the M2 pivot mechanism. | NO JOURNAL ENTRY | live | KEEP | imported by 2 python module(s) |
| `scale/merkle.py` | A Merkle journal, so "the record was not edited" becomes checkable. | NO JOURNAL ENTRY | live | KEEP | imported by 2 python module(s) |
| `scale/monge.py` | Exact rectangular assignment for a Monge cost on a line, as an oracle. | NO JOURNAL ENTRY | live | KEEP | imported by 2 python module(s) |
| `scale/negation_scope.py` | M3 — long-range sign capability, with an ABSOLUTE bar. | 9/54 readings reproduce at abs=5e-7 | live | KEEP | AMENDED r10-it2: the iteration-1 cell read "no python importer"; re-measured with a front-door grep there are 45 (scale/capability_table.py, scale/e_ladder.py, scale/eprocess_perdraw.py and 42 more). Disposition unchanged. no python importer, but 9/54 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/orbit_probe.py` | caustic Theorem 2, pointed at attention rows. | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers, 0 results artifacts, 0 prose citations in any tracked file. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/page_trend.py` | LADDER E's "the curve rises" verdict, as a named statistic instead of an eye. | results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | pre-seeded KEEP (proven this branch): 51 achievable p-values at k=4 N=5, critical L=137, exact size 0.037002877 |
| `scale/paired_arm.py` | Per-example eval predictions for one arm, so sign-vs-routing gets the RIGHT test. | NO JOURNAL ENTRY | live | KEEP | imported by 5 python module(s) |
| `scale/pivot_probe.py` | M2 — context-stable signed influence at GLOBAL reach, via pivot routing. | 2/3 readings reproduce at abs=5e-7 | live | KEEP | AMENDED r10-it2: the iteration-1 cell read "no python importer"; re-measured with a front-door grep there are 52 (scale/aggregator_matched_filler.py, scale/aggregator_mechanism.py, scale/arm_a_k1.py and 49 more). Disposition unchanged. no python importer, but 2/3 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/pivot_selection_theory.py` | Two facts about pivot selection that this repository's numbers rest on. | NO JOURNAL ENTRY | live | KEEP | imported by 1 python module(s) |
| `scale/r2_units.py` | R2 -- sign-determinacy of the k x k pivot block, and its replacement. | results/r2.jsonl (36 rec) via Journal(NAME="r2") -> scale/bucket.py:45 | live | KEEP | producer of results/r2.jsonl - path CONSTRUCTED via Journal(NAME="r2") -> scale/bucket.py:45, invisible to a literal scan |
| `scale/r4b_units.py` | R4b — hierarchical (Dyson) coupling wired into the REAL operator. | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers, 0 results artifacts, 0 prose citations in any tracked file. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/ratio_sweep.py` | Sweep the sgate-vs-ParaFormer sign-flip separation across context length. | NO JOURNAL ENTRY | live | KEEP | imported by 1 python module(s) |
| `scale/recall_probe.py` | Dr House's experiment: is it routing, or a shrunken denominator? | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 6 tracked file(s) (.superpowers/sdd/polymorphic-drifting-squirrel/progress.md, DONE_ARCHIVE_ROUND1.md), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/replay_census.py` | Replay EVERY journalled m2 unit, not the one the rotation happened to pick. | results/m2.jsonl (40 rec) | live | KEEP | producer of results/m2.jsonl |
| `scale/rip_line.py` | The compressed-sensing floor under IMPACT's attribution column. | NO JOURNAL ENTRY | live | KEEP | imported by 1 python module(s) |
| `scale/rips_gate.py` | The E4 admission gates: does the Rips connectivity label need an iteration? | NO JOURNAL ENTRY | live | KEEP | imported by 7 python module(s) |
| `scale/route_dependency.py` | M2' ROUTE-DEPENDENCY PROBE â€” are R1/R2/R3/R4 four routes or one mechanism? | NO JOURNAL ENTRY | live | KEEP | imported by 1 python module(s) |
| `scale/run_m2.py` | M2, per CHECKLIST.md, verbatim. LOCK M2 efadc390c93f. | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 1 tracked file(s) (DONE_ARCHIVE_ROUND1.md), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/s2_probe.py` | S2 — the selection ablation, run standalone so it cannot collide with the loop. | results/m2.jsonl (40 rec) | live | KEEP | producer of results/m2.jsonl |
| `scale/s2_units.py` | S2 — the ablation that decides whether the contribution is ROUTING or SIGNEDNESS. | results/s2.jsonl (13 rec) via Journal(NAME="s2") -> scale/bucket.py:45 | live | KEEP | producer of results/s2.jsonl - path CONSTRUCTED via Journal(NAME="s2") -> scale/bucket.py:45, invisible to a literal scan |
| `scale/scale_sweep.py` | Does the parity ratio hold as the model grows, or is 3.3M an artifact? | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 3 tracked file(s) (house-events.jsonl, tests/chase/axes.py), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/settle.py` | The settling driver: iterate T to its fixed point, journal the Hilbert residual. | NO JOURNAL ENTRY | live | KEEP | imported by 3 python module(s) |
| `scale/sgate_softmax_probe.py` | TWO QUESTIONS ABOUT `_causal_sgate_operator`, MEASURED. | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers, 0 results artifacts, 0 prose citations in any tracked file. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/sign_dependence_probe.py` | IS A_8 = 2.187500 A VALID CERTIFICATE? Test the sign-independence assumption. | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 1 tracked file(s) (DONE.md), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/sparse_probe.py` | Does a bounded-reachability schedule arrest the 1/s decay? | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 4 tracked file(s) (.superpowers/sdd/polymorphic-drifting-squirrel/venus-report-it3.md, DONE_ARCHIVE_ROUND1.md), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/sprt.py` | Wald's sequential probability ratio test, and the K1 slope-to-rate mapping. | 1/1 readings reproduce at abs=5e-7 | live | KEEP | AMENDED r10-it2: the iteration-1 cell read "no python importer"; re-measured with a front-door grep there are 1 (tests/chase/test_sprt_k1.py). Disposition unchanged. no python importer, but 1/1 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/step_profile.py` | WHERE DOES ONE FULL-BATCH TRAINING STEP OF `run_arm` ACTUALLY GO? | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 1 tracked file(s) (scale/sgate_softmax_probe.py), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/tau_trajectory.py` | X6, second half -- the //tau//_F TRAJECTORY, which is what the clause names. | results/equilibrium.jsonl (9 rec); results/tau_trajectory.jsonl (10 rec) | live | KEEP | producer of results/tau_trajectory.jsonl |
| `scale/tgate_probe.py` | Ladder step 1: does the denominator-free operator hold its property in context? | 1/3 readings reproduce at abs=5e-7 | live | KEEP | no python importer, but 1/3 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/torque_probe.py` | ARM A — the torque probe. Pure measurement. Nothing is built until this survives. | NO JOURNAL ENTRY | live | KEEP | imported by 22 python module(s) |
| `scale/trained_projections.py` | Phase D: every probe number in this project was taken at random init. This is the delta. | NO JOURNAL ENTRY | live | KEEP | imported by 2 python module(s) |
| `scale/twodof.py` | The 2-dof lemma, and the two loci where it fails (contract 1.3). | 7/7 readings reproduce at abs=5e-7 | live | KEEP | AMENDED r10-it2: the iteration-1 cell read "no python importer"; re-measured with a front-door grep there are 2 (scale/trained_projections.py, tests/cameron/test_r6_twodof_red.py). Disposition unchanged. no python importer, but 7/7 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/valuation.py` | X4 — the sign-flip statistic on VALUATIONS, not floats. The floor is DELETED. | 1/2 readings reproduce at abs=5e-7 | live | KEEP | AMENDED r10-it2: the iteration-1 cell read "no python importer"; re-measured with a front-door grep there are 6 (scale/arm_a_k1.py, scale/arm_a_rebuild.py, scale/arm_a_run.py and 3 more). Disposition unchanged. no python importer, but 1/2 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/vgpe.py` | VGPE -- verb-gauge positional encoding, and the K1 gate that decides it. | 4/8 readings reproduce at abs=5e-7 | live | KEEP | AMENDED r10-it2: the iteration-1 cell read "no python importer"; re-measured with a front-door grep there are 1 (tests/watson/test_vgpe_binds.py). Disposition unchanged. no python importer, but 4/8 of its readings reproduce from results/*.jsonl at abs=5e-7 |
| `scale/vgpe_flops.py` | Analytic FLOP accounting for the five V12 gauge-positional-encoding arms. | results/arm_s.jsonl (82 rec) | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 1 tracked file(s) (V12_PRICING.md), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |
| `scale/wilson_g2_ci_probe.py` | WILSON -- G2 event: the D_FR slope bootstrap interval that moved. | results/arm_a_k1.jsonl (48 rec) | live | KEEP | producer of results/arm_a_k1.jsonl |
| `scale/wilson_probes.py` | WILSON -- verification probes for the fellows' claims. Pure measurement. | NO JOURNAL ENTRY | live | KEEP | imported by 1 python module(s) |
| `scale/window_sweep.py` | Does a bounded receptive field arrest the context decay, and for whom? | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: 0 python importers and 0 results artifacts. Named in prose by 2 tracked file(s) (tests/cameron/test_composition_is_the_uncosted_route.py, tests/foreman/test_window_width_is_the_missing_control.py), so a document rests on a producer nothing runs. PRESUMPTION, NOT VERDICT; see KILL 3 |

### root *.md

Count: **53** &mdash; `git ls-files -- '*.md' | grep -v /`

| path | claims to test | last-run state FROM THE JOURNAL | class | KEEP/ATTIC | one-line reason |
|---|---|---|---|---|---|
| `ADR-001-bucketed-measurement.md` | ADR-001: Bucketed, resumable measurement units | NO JOURNAL ENTRY | orphan | ATTIC | PRESUMED orphan: no other tracked file names this document or its stem (measured over 516 tracked text files). PRESUMPTION, NOT VERDICT; see KILL 4 |
| `ARSENAL.md` | CEQ MATHEMATICAL ARSENAL — post-audit consolidation, v2 | NO JOURNAL ENTRY | live | KEEP | named by 14 tracked file(s) |
| `BACKLOG.md` | Backlog — the parity campaign | NO JOURNAL ENTRY | live | KEEP | named by 5 tracked file(s) |
| `BOARD.md` | THE ENGINEERING BOARD — nurse findings, for Wilson | results/m3_quintuple_v2_cuda.jsonl (23 rec) | live | KEEP | named by 20 tracked file(s) |
| `CHECKLIST.md` | CEQ-NOVELTY CHECKLIST — WORK-STOPPING CLAUSE | results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | named by 63 tracked file(s) |
| `CONTRACT.md` | THE CONTRACT | NO JOURNAL ENTRY | live | KEEP | named by 26 tracked file(s) |
| `D1.md` | D1 — The negative result | results/m3_quintuple_v2.jsonl (151 rec); results/m3_quintuple_v2_cuda.jsonl (23 rec) | live | KEEP | named by 32 tracked file(s) |
| `DONE.md` | HILBERT: IN PROGRESS - round 7, CEQ v9, THE CAPABILITY ROUND, iteration 0 of 30 | results/arm_a.jsonl (3 rec); results/arm_a_k1.jsonl (48 rec) +10 | live | KEEP | named by 71 tracked file(s) |
| `DONE_ARCHIVE_ROUND1.md` | Done | results/m2.jsonl (40 rec) | superseded | KEEP | self-labelled round archive; its successor names it as superseded. Retained: it is the provenance of published readings |
| `E_LADDER_PREREGISTERED_READING.md` | Ladder E — the settled−twin dose-response, pre-registered before the run | results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | named by 16 tracked file(s) |
| `HOUSE_BRIEF.md` | BRIEF FOR DR HOUSE — the whole state, and the honest accounting | results/hilbert.jsonl (142 rec) | live | KEEP | named by 1 tracked file(s) |
| `IMPOSSIBLE.md` | DR HOUSE TODO — the impossibility list | NO JOURNAL ENTRY | live | KEEP | named by 6 tracked file(s) |
| `LOOP.md` | Loop protocol — consequence-equilibrium attention | NO JOURNAL ENTRY | live | KEEP | named by 75 tracked file(s) |
| `LOOP_PROMPT.md` | CEQ v10 — ROUND 8: THE EQUILIBRIUM-CORPUS ROUND | results/foreman_looped.jsonl (4 rec) | live | KEEP | named by 62 tracked file(s) |
| `LOOP_PROMPT_ROUND2_ARCHIVE.md` | THE LOOP PROMPT — read this file in full, every iteration, and follow it exactly | 14/33 readings reproduce at abs=5e-7 | superseded | KEEP | self-labelled round archive; its successor names it as superseded. Retained: it is the provenance of published readings |
| `LOOP_PROMPT_ROUND3_ARCHIVE.md` | CEQ v5 — ROUND 3 LOOP PROMPT. Read in full, every iteration, follow exactly. | 3/4 readings reproduce at abs=5e-7 | superseded | KEEP | self-labelled round archive; its successor names it as superseded. Retained: it is the provenance of published readings |
| `LOOP_PROMPT_ROUND4_ARCHIVE.md` | CEQ v6′ — ROUND 4 LOOP PROMPT. The chosen-sign round. Read in full, follow exactly. | 7/14 readings reproduce at abs=5e-7 | superseded | KEEP | self-labelled round archive; its successor names it as superseded. Retained: it is the provenance of published readings |
| `LOOP_PROMPT_ROUND5_ARCHIVE.md` | CEQ v7 — ROUND 5. THE TWO-SPHERES ROUND. Read in full, follow exactly. | 12/26 readings reproduce at abs=5e-7 | superseded | KEEP | self-labelled round archive; its successor names it as superseded. Retained: it is the provenance of published readings |
| `LOOP_PROMPT_ROUND6_ARCHIVE.md` | CEQ v8.2 — ROUND 6. THE HILBERT ROUND. Read in full, follow exactly. | 8/23 readings reproduce at abs=5e-7 | superseded | KEEP | self-labelled round archive; its successor names it as superseded. Retained: it is the provenance of published readings |
| `LOOP_PROMPT_ROUND7_ARCHIVE.md` | CEQ v9 — ROUND 7. THE CAPABILITY ROUND. Read in full, follow exactly. | 0/2 readings reproduce at abs=5e-7 (was recorded 1/3) | superseded | KEEP | AMENDED r10-it2, NOT a drawn row — found by generalising the workdone2.md check to every row invoking the provenance ground. Re-measured with the extractor calibrated on this sheet's own control: 0 of 2 readings reproduce from any results/*.jsonl at abs=5e-7. The provenance ground is EMPTY, the same defect as `workdone2.md`. KEEP stands on a different, measured ground: named by 1 tracked file (STATE.md). Kept as a cited round archive, NOT as provenance. |
| `M2PRIME_PREREGISTERED_READING.md` | M2′ R2 — what each outcome will mean, written BEFORE the run | NO JOURNAL ENTRY | live | KEEP | named by 4 tracked file(s) |
| `M2_PREREGISTERED_READING.md` | M2 — what each outcome will mean, written BEFORE the control finished | 10/12 readings reproduce at abs=5e-7 | live | KEEP | named by 7 tracked file(s) |
| `M2_TRAINED_PREREGISTERED_READING.md` | A trained measurement — what each outcome will mean, written BEFORE any run | 1/9 readings reproduce at abs=5e-7 | live | KEEP | named by 8 tracked file(s) |
| `M3_QUINTUPLE_PREREGISTERED_READING.md` | M3 — the deciding measurement, pre-registered before the run | results/m3_quintuple.jsonl (19 rec) | live | KEEP | named by 23 tracked file(s) |
| `MATHEMATICS.md` | MATHEMATICS.md — the theory of record | results/hilbert.jsonl (142 rec); results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | named by 12 tracked file(s) |
| `METHODS.md` | METHODS — the anytime-valid decision statistic | NO JOURNAL ENTRY | live | KEEP | named by 8 tracked file(s) |
| `MISTAKES.md` | MISTAKES — the failure taxonomy of this repository | results/foreman_looped.jsonl (4 rec); results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | pre-seeded KEEP (proven this branch): the failure taxonomy of record, classes V/P/M/D |
| `MODEL_CARD.md` | --- | results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | named by 16 tracked file(s) |
| `NOTES.md` | Cutting edge — scan of github.com/teerthsharma, 2026-08-24 | NO JOURNAL ENTRY | live | KEEP | named by 1 tracked file(s) |
| `PIVOT_EXCLUSION_FALSIFIER.md` | The pivot-exclusion confound, and the falsifier that decides it | 4/21 readings reproduce at abs=5e-7 | live | KEEP | named by 8 tracked file(s) |
| `PREREGISTRATION_HOLE_AUDIT.md` | Pre-registration hole audit — all five documents, before R9 | results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | named by 7 tracked file(s) |
| `PRIOR_ART.md` | PRIOR ART — the Hankel program's citation of record | 1/46 readings reproduce at abs=5e-7 | live | KEEP | named by 8 tracked file(s) |
| `PROGNOSIS.md` | Prognosis | 3/24 readings reproduce at abs=5e-7 | live | KEEP | named by 11 tracked file(s) |
| `R9_IRENE_PREDICTION.md` | R9 — the competing prediction, filed before any R9 number existed | 7/41 readings reproduce at abs=5e-7 | live | KEEP | named by 9 tracked file(s) |
| `README.md` | ceq — pivot-routed attention with an optional fixed-point settle | results/m3_quintuple_v2.jsonl (151 rec) | live | KEEP | named by 55 tracked file(s) |
| `REQUIREMENTS.md` | The module — six behavioral requirements | 1/2 readings reproduce at abs=5e-7 | live | KEEP | named by 16 tracked file(s) |
| `RESEARCH.md` | CEQ: Content-Conditional Sign in Multi-Hop Attention — A Negative Result | 10/33 readings reproduce at abs=5e-7 | live | KEEP | named by 9 tracked file(s) |
| `STATE.md` | STATE — CEQ v10, ROUND 8, THE EQUILIBRIUM-CORPUS ROUND | 11/46 readings reproduce at abs=5e-7 | live | KEEP | named by 69 tracked file(s) |
| `THEORY.md` | Consequence-Equilibrium Attention — Theory v1 | NO JOURNAL ENTRY | live | KEEP | named by 42 tracked file(s) |
| `THEORY2.md` | Theory 2 — DEAD. All three branches, killed by the room. | 3/5 readings reproduce at abs=5e-7 | live | KEEP | named by 3 tracked file(s) |
| `THEORY_V12_VGPE.md` | THEORY_V12_VGPE.md — VGPE as a weighted automaton, and what its capacity actually is | 11/49 readings reproduce at abs=5e-7 | live | KEEP | named by 3 tracked file(s) |
| `TRAINING.md` | TRAINING.md — training this model on free compute | NO JOURNAL ENTRY | live | KEEP | named by 17 tracked file(s) |
| `V12_IRENE_FILING.md` | V12 — the competing prediction on VGPE, filed before any V12 cell has run | 1/14 readings reproduce at abs=5e-7 | live | KEEP | named by 1 tracked file(s) |
| `V12_MORIARTY_CONTROLS.md` | V12 — the adversarial controls on VGPE, filed before any V12 cell has run | 1/14 readings reproduce at abs=5e-7 | live | KEEP | named by 1 tracked file(s) |
| `V12_PRICING.md` | V12 PRICING — the five gauge-positional arms, priced before the round runs | NO JOURNAL ENTRY | live | KEEP | named by 2 tracked file(s) |
| `done3.md` | Work done — round 3 (CEQ v5) | 10/48 readings reproduce at abs=5e-7 | superseded | KEEP | per-round work log closed by a later round's log; retained as the provenance of its round's readings |
| `done4.md` | Work done — round 4 (CEQ v6′), the chosen-sign round | 7/16 readings reproduce at abs=5e-7 | superseded | KEEP | per-round work log closed by a later round's log; retained as the provenance of its round's readings |
| `done5.md` | done5 — Round 5, CEQ v7, the two-spheres round | results/arm_a.jsonl (3 rec) | superseded | KEEP | per-round work log closed by a later round's log; retained as the provenance of its round's readings |
| `done6.md` | done6 — Round 6, CEQ v8.2, the Hilbert round | 13/39 readings reproduce at abs=5e-7 | superseded | KEEP | per-round work log closed by a later round's log; retained as the provenance of its round's readings |
| `done7.md` | done7.md — CEQ v9, round 7, handoff at iteration 11 | results/m3_quintuple_v2.jsonl (151 rec) | superseded | KEEP | per-round work log closed by a later round's log; retained as the provenance of its round's readings |
| `titan-report.md` | TITAN report — D-1, D-2, H-1 partial-ladder guards on `scale/e_ladder.py::verdict()` | results/m3_quintuple_v2.jsonl (151 rec) | orphan | ATTIC | PRESUMED orphan: no other tracked file names this document or its stem (measured over 516 tracked text files). PRESUMPTION, NOT VERDICT; see KILL 4 |
| `workdone2.md` | Work done — round 2, iterations 34–46 | 0/4 readings reproduce at abs=5e-7 (was recorded 1/10) | superseded | KEEP | AMENDED r10-it2, THE ONE KEEP FAILURE OF THE BINOMIAL. The iteration-1 ground was "provenance of published readings" on a cell of 1/10. Re-measured with the extractor CALIBRATED against this sheet's own control (LOOP_PROMPT_ROUND6_ARCHIVE.md re-measures 8/23, exactly its cell): 0 of 4 readings reproduce from any results/*.jsonl at abs=5e-7. Nearest journal values miss by 1.6e-4 to 35.5, i.e. 300x to 7e7x the tolerance -- near-misses, not a parsing artifact. The provenance ground is EMPTY. KEEP stands on a DIFFERENT, measured ground: 7 tracked files name it (DONE.md, done3-6.md, scale/chase_struck_coverage.py, tests/cameron/test_composition_is_the_uncosted_route.py). Kept as a cited round log, NOT as provenance. |
| `workdoneplanetrum.md` | workdoneplanetrum | 12/41 readings reproduce at abs=5e-7 | superseded | KEEP | per-round work log closed by a later round's log; retained as the provenance of its round's readings |

### round reports misfiled under tests/ (audited as docs, not tests)

Count: **4** &mdash; `git ls-files -- 'tests/' | grep '\.md$'`

These are root-doc-class rows sitting in the test tree. Counting them as tests over-reports the test denominator by 4; counting them nowhere under-reports the doc denominator by 4.

| path | claims to test | last-run state FROM THE JOURNAL | class | KEEP/ATTIC | one-line reason |
|---|---|---|---|---|---|
| `tests/deimos/DEIMOS_REPORT.md` | DEIMOS report — R9 iteration 1 | NO JOURNAL ENTRY | live | KEEP | in-tree artifact named by 4 tracked file(s) |
| `tests/mars/MARS_REPORT_IT2.md` | MARS / MORIARTY — R9 iteration 2 | 4/13 readings reproduce at abs=5e-7 | live | KEEP | in-tree artifact named by 1 tracked file(s) |
| `tests/mars/MARS_REPORT_IT3.md` | MARS / MORIARTY — R9 iteration 3, the restitution round | NO JOURNAL ENTRY | orphan | ATTIC | in-tree report artifact named by 0 other tracked files (measured over 516 tracked text files); see KILL 4 |
| `tests/mars/MARS_REPORT_IT4.md` | MARS / MORIARTY — R9 iteration 4: the three GREENs that had no adversary | 6/18 readings reproduce at abs=5e-7 | orphan | ATTIC | in-tree report artifact named by 0 other tracked files (measured over 516 tracked text files); see KILL 4 |

## North Star

> "A consequence-understanding, path-understanding, causality-understanding attention architecture, as
> good as self-attention on a calibrated bar, that predicts the NEXT BEST ACTION toward equilibrium &mdash;
> not the next best token. The novelty is the WHOLE ARCHITECTURE; components may be pre-existing, always cited."

**Distance.** This sheet moves the North Star by zero and was not meant to: it is a census, and its whole
output is knowing which of 347 rows can still carry a claim. The one thing it changes about the distance is
that 40 rows are now known not to be evidence, so the calibrated bar is measured over what is left rather
than over the whole tree.
