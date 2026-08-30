# R10 — v-main.3M Iteration 1 — P0.1, the census

Phase 0, TRIAGE. Room dispatched in parallel: SATURN (sheet), NEPTUNE (factual
baseline / systems gate), MARS (standing adversary). Health Inspector audited the
log before this record was written.

Deliverable: `AUDIT.md`, 638 lines, 349 table rows = 347 classed + 2 journals
listed and unclassed per L-G2.

## Verdict

The sheet ships. Its classes are not yet trustworthy, and the census measured why:
four class presumptions entered, three of them failed, and all three failed the same
way — each keyed on a surface feature of a file instead of the role the instrument
plays. That pattern, not any individual row, is the iteration's result.

## Final counts

Recounted independently from the shipped file, not from any agent's generator:

| class | n | disposition |
|---|---|---|
| live | 292 | KEEP |
| superseded | 14 | KEEP |
| struck | 1 | KEEP |
| orphan | 25 | ATTIC |
| vacuous | 15 | ATTIC |
| **classed rows** | **347** | **KEEP 307 / ATTIC 40** |

Presumed rows: 36. NO JOURNAL ENTRY: 198 of 347.

Commands of record:

```
git ls-files -- 'tests/*.py'  | wc -l          ->  191
git ls-files -- 'scale/*.py'  | wc -l          ->   99
git ls-files -- '*.md' | grep -v / | wc -l     ->   53
                                               +    4   round reports misfiled under tests/
                                               =  347
```

Trajectory across SATURN's three self-corrections: ATTIC 52 -> 47 -> 40, vacuous
27 -> 22 -> 15, presumed 48 -> 43 -> 36. Every correction moved rows out of ATTIC.
The instrument over-condemned in one direction only.

## The pattern — four presumptions, three deaths, one mechanism

| # | presumption | keyed on | should key on | outcome |
|---|---|---|---|---|
| P1 | control constructs own input => vacuous | *who* built the input | what path the input takes | KILLED by MARS |
| P3-literal | literal path absent => no journal => orphan | a literal path string | the constructed path | KILLED by SATURN, FINDING 1 |
| P1' | plant enters below the front door => vacuous | *where* input enters | the instrument's claim | KILLED by SATURN, FINDINGS 2 and 3 |
| P2/P3-orphan | zero importers AND zero results AND zero prose citations | three-part conjunction | — | SURVIVED audit |

Each dead rule fired on method and called it defect. The only thing that caught it
each time was running the code rather than reading it.

### FINDING 1 (SATURN) — bound, RED still failing on the shipped tree

`scale/bucket.py:45` builds the journal path as `RESULTS / f"{name}.jsonl"`. The path
is constructed and never written down, so a literal-path census reports NO JOURNAL
ENTRY and rule P3 then presumes live producers dead. It did: SATURN's first pass
classed `scale/s2_units.py` ORPHAN/ATTIC. It produces `results/s2.jsonl`, 13 records,
13 of 15 six-significant-digit readings reproducing at abs=5e-7. Same error on
`scale/r2_units.py` -> `results/r2.jsonl`, 36 records.

This is MISTAKES.md **V-7** — the exact defect `scale/journal_scan.py` was written to
abolish — reintroduced one level up, in the code that chooses which files to hand it.

RED: `tests/saturn/test_journal_path_is_discoverable.py`, 3 failed / 3 passed.
Re-run by the Inspector: 3 failed / 3 passed in 1.71s, failing on m2/r2/s2_units.
Deliberately left RED: it names an unrepaired defect in `scale/`, and repairing
`scale/` is not the census seat.

### FINDING 2 (SATURN) — the prior-art defence was five minutes from being filed dead

`tests/foreman/test_signedness_is_not_new.py` reimplements SimA (arXiv 2206.08898)
and Differential Transformer (arXiv 2410.05258) from their defining equations and
races them on the module's own instrument. It answers whether signedness is new at
all. P1' classed it vacuous / ATTIC.

A refutation instrument constructs its own input *by necessity*: the mechanism it
races belongs to a competitor, and a competitor has no production batch path in this
repo. Building SimA from its paper is the method, not a shortcut around one.

`grep -rln "def test_claim_" tests/` -> 12 files. The sheet had ATTIC'd 5.

### FINDING 3 (SATURN) — a live rejection region withdraws the vacuity presumption

Sweep over the 22 condemned files: 61 failed, 197 passed, 34 xfailed in 372.16s.
22 failures under `test_claim_*` (the 5 files FINDING 2 corrected); 39 under other
names across 7 more files still classed vacuous.

A test that is currently rejecting is not passing by construction, so it cannot be
vacuous in the strong sense. P1' is worded *until shown otherwise*; a live rejection
region shows otherwise. All 12 moved to live/KEEP.

What a RED does not settle: whether the scope is right. A test can fail for a reason
unrelated to its claim. Narrowed, not closed. The 10 files that ran all-green keep
their presumption — passing is precisely what a vacuous control does.

`tests/foreman/conftest.py` is a `sys.path` shim only; the `xfail(strict=True)`
`KNOWN_RED` ledger at `tests/chase/conftest.py:285` is scoped to `tests/chase/`.
Foreman's reds are unledgered by design, because for those files RED is the
deliverable. `DONE_ARCHIVE_ROUND1.md:1013` lists exactly these names under `RED:`.

### MARS's attack — fires

Target: P1 misfires on `tests/cameron/test_impact_hetero_is_not_its_own_baseline.py`,
the file binding MISTAKES.md class V-1. It draws every asserted-on tensor through
`NS.M3_TASKS[name][0](...)`. Under P1 that is vacuous-by-scope, therefore ATTIC.

Settling measurement: restore the V-1 defect at `scale/negation_scope.py:1351`
(`make_impact_hetero_batch` -> `make_impact_batch`) and re-run.
Shipped registry: 6 passed in 31.92s. Defect restored: 4 failed, 2 passed in 35.46s,
`assert 2.289092002991096e-08 > 1.0`. The control has a rejection region.

Mechanism: P1 keys on who constructed the input. V-14 in MISTAKES.md keys on the path
the input takes — *"a planted positive must enter at the instrument's front door and
traverse every stage the real input traverses."* Every planted positive is
constructed; that is what planting means. P1 as written has no discriminating power
and fires on cures as readily as diseases.

Replacement shipped (reroute + reprice). P1' has a demonstrated rejection region:
it fires on `scale/chase_struck_coverage.py`, whose `control()` at `:105` calls
`scan_text()` on literals at `:107`/`:109` and never `collect_targets()`, reached only
at `:137`, while `collect_targets()` reaches 409 paths. Matcher exercised, reach never
— the V-14 shape, still in the tree today.

### NEPTUNE's finding — 56 tests are green only on this machine

`tests/cameron/test_capability_result.py`: 10 passed in the working tree; 2 failed,
8 passed on a clean clone of the same commit. `FileNotFoundError:
data/scan_addprim_jump_train.txt missing` at `ceq/harness.py:159`.

`git ls-files data/` -> 1 file (`data/README.md`). `ls data/` -> 12. The 11 corpora
(SCAN, COGS, TinyStories) are gitignored deliberately and with citations per
`data/README.md`, but no tracked runner fetches them. `.gitignore:29-30` excludes
`data/*.tsv` and `data/*.txt`.

Blast radius, measured: 17 corpus-dependent test files in the clean clone —
56 failed, 177 passed, 31 skipped, 4 xfailed, 61.50s.

## Systems ground truth (NEPTUNE)

- Importer graph, AST over 102 tracked `.py`, zero parse failures. 123 targets
  (99 `scale/*.py` + 24 `ceq/`; `ceq/hf_artifact/` excluded as a published-artifact
  duplicate). **78 have >= 1 real in-scope importer, 45 have zero** — of those, 19
  write a results artefact and 26 do not.
- Results linkage, 242 files: 44 named by literal basename in a producer, 159 via
  `NAME`-const -> `bucket.Journal`/`WEIGHTS_DIR`, 4 by parent dir, **32 with no
  producing code**. Of the 32, 20 are named only in prose and **12 are named nowhere
  in any tracked file**.
- **The repo tracks exactly two runners: `pytest.ini` and `colab/train_ceq.ipynb`.**
  No Makefile, no CI, no shell script. 33 of the 37 `.txt` files carry a module stem
  but no code writes them — they are stdout captures whose shell is not committed.
- Collection: 2071 tests, zero errors, identical in a fresh clone.
- Module-level skips: exactly 1, `tests/cameron/test_harmonic_attribution.py`
  (`pytest.skip(` opened at `:176`, `allow_module_level=True` at `:188`).
- Full-suite price, ESTIMATE from an 18% CPU-only sample measured this iteration:
  376 passed in 111.13s = 0.2956 s/test x 2071 = **612s ~ 10.2 min**. A floor, not a
  band — the sample excludes dirs carrying model downloads and training.
- Cold path: clone 1.324s / 98 MB / 681 files. clone + one test = **4.99s torch-free,
  5.74s with torch**. Install term unmeasured: 75 packages resolved, torch occupies
  4.4 GB on disk in this environment.

## Answers the census was asked for

- **STRUCK registry** is not a file. No `STRUCK.md` exists. It is
  `tests/loop/test_no_struck_constant_ships.py:47`, a module-level
  `STRUCK: dict[float, str]`, asserted non-empty by `inspector.py:459`.
- **11/11 AttributeError family**: one file, eleven tests —
  `tests/cameron/test_harmonic_attribution.py`, 411 lines of which 302 are the
  verbatim record. Module-level skip guarded by a live `hasattr` check at `:170-174`,
  so it un-skips itself when a producer appears.
- **Journal contradicting a file: zero rows** at abs=5e-7. What exists instead is
  absence: of 8,519 six-significant-digit literals across 181 files, 1,900 reproduce
  from `results/*.jsonl` and 5,676 appear in no `results/` artefact of any extension.
  That is an exposure surface, not 5,676 findings, and it is not priced here.
- **Journal denominator**: `results/` holds 242 tracked files — 165 `.pt`, 37 `.txt`,
  27 `.jsonl`, 7 `.json`, 6 `.md`. Only **27 are journals**, holding 809 records.

## Health Inspector — 20 claims audited, 3 struck

**Strike 1, SATURN.** The `done` event at `house-events.jsonl:9054` records
`live:280, vacuous:27, keep:295, attic:52, presumed:48`. `AUDIT.md:50-62` records
`live 285 | vacuous 22 | KEEP 300 | ATTIC 47 | presumed 43`. Same total, five-row
shift in every split. The log event is struck; **AUDIT.md is the artifact of record.**

**Strike 2, MARS.** The finding at `house-events.jsonl:8799` claims `impact_hetero`
goes RED 4/6 under the V-1 revert. No `status:red` event precedes it — the mutation
run was never logged as a test event. It was retroactively bound 14 lines later at
`:8813`. The finding as posted carried no RED.

**Strike 3, the log.** 18 distinct GREEN `t:test` events name a node that cannot be
re-run. 13 carry no file path at all — the `name` field is prose
(`sprt_k1_mapping`, `run_calib.py --self-test`, `it.0 Hankel instrument: 36 passed`).
The other 5 name a real file with counts appended, so the string is not a node id.
These are historical, from earlier rounds, not from this iteration — but they are
exactly the failure mode the log exists to prevent.

**Bound/unbound**: all three of this iteration's findings are BOUND. SATURN's
FINDING 1 is the strongest — its RED still fails on the shipped tree today.

**Collection reconciles exactly.** NEPTUNE measured 2071; the Inspector measured
2094. The delta of 23 is precisely the three untracked test files added during this
iteration. 2094 - 23 = 2071. Clean, not drift.

## Corrections to the record

**House was wrong about the orphan class.** This record's author stated twice that
NEPTUNE's `__main__` fact kills 19 of SATURN's 25 orphan rows and that the orphan
class was unrepaired. Overruled by audit: `AUDIT.md:112` states P3 as *zero importers
AND zero results*, and `AUDIT.md:225-226` state a three-part conjunction adding *zero
prose citations*. `__main__`-reachability is not a conjunct of any of the three. The
19 rows stand. The claim asserted what the class rested on without reading how the
sheet stated it.

**NEPTUNE's factual baseline carried one false claim.** He reported *"Only 3 files in
the whole repo have no `__main__` token at all: `scale/window_sweep.py`,
`scale/sign_dependence_probe.py`, `scale/replay_census.py`."* Measured: **16
`scale/*.py` files carry no `__main__`**, including `kirchhoff.py`,
`identity_manifest.py`, `journal_scan.py`, `bucket.py`, `merkle.py`, `settle.py` —
library modules, which have no `__main__` because they are libraries. His three are a
subset of the sixteen. His scoped finding (24 of the 26 zero-importer/zero-results
files are `__main__`-reachable) is unaffected; the "whole repo" generalisation is
false and is withdrawn.

**Two briefs from House carried wrong denominators**, both caught by the room:
`197` test files (correct: 191 `.py`; the other 6 are 2 journals and 4 misfiled round
reports) and `242 journals` (correct: 242 artefacts, of which 27 are journals).

## Open

1. **Three shipped numbers have no recorded file list.** "Files reaching a production
   builder" is claimed at three widths — MARS 22, SATURN 46, nurse census 123 — and
   nesting can be neither shown nor refuted, because no set was recorded behind any of
   them. `AUDIT.md:100` cites `scratchpad/p1prime.py` as provenance for the 46. That
   file exists — in the *session temporary directory*, which dies with the session.
   NEPTUNE's `scratchpad/coldclone` is likewise a real clone at
   `3e98563025ffa1239f7f14136645321fedb4e892`, and likewise ephemeral. **The
   measurements were real and are about to become unreproducible.** Provenance for a
   shipped number must land in the repo or in `results/`, never in temp.
2. **Three untracked test files carry live claims**:
   `tests/mars/test_p1_presumption_overfires.py` and both `tests/saturn/*.py`. An
   untracked RED is invisible to CI and to every clone — the same defect class as the
   `data/` finding.
3. **Three unpriced failures against controls the module must beat**, surfaced by the
   census sweep and routed out of the census seat unanswered:
   `tests/foreman/test_r1_settling.py:61`
   `test_the_stated_R1_falsifier_rejects_the_affine_update_it_is_meant_to_kill`
   **fails** — a must-fire arm not firing (V-10 shape); 10 failures against APPNP, the
   single control REQUIREMENTS.md R2 names as must-beat; 10 failures on 300M parity
   survival. Whether these are refutations already priced in an earlier round or
   regressions on this branch is unanswered, and it determines whether the calibrated
   bar is where the KEEP column suggests.
4. **The 40-iteration script's it.4 cannot run as written.** It specifies
   `make verify`. There is no Makefile; the repo tracks two runners, neither is `make`.
   Either `make verify` is built during P0.4 layout or it.4 is rewritten to name the
   actual entry points.
5. **The 36 presumed rows remain presumed.** They are what iteration 2's binomial
   spot-check must cover.
6. Five `tests/chase` files produce collection errors as a set while collecting clean
   alone — name collision or import-order interaction, unexplained.
7. `scale/replay_census.py` writes at module scope (`OUT.open("w")`) with no
   `__main__` guard; the write fires on import. Nothing imports it, so it is inert.
   No test binds this.
8. `scale/vgpe_flops.py` is the sole producer behind 377 lines of `V12_PRICING.md`
   and writes no replayable artefact.

## North Star

> "A consequence-understanding, path-understanding, causality-understanding attention
> architecture, as good as self-attention on a calibrated bar, that predicts the NEXT
> BEST ACTION toward equilibrium — not the next best token. The novelty is the WHOLE
> ARCHITECTURE; components may be pre-existing, always cited."

**Distance: zero metres, by design.** A census moves nothing toward the architecture.
What it changed is that 40 rows are now known not to be evidence, 12 rows that were
about to be filed as dead weight are back — including the prior-art defence of the
whole contribution — and three failures against named must-beat controls are now
visible instead of buried in a sweep nobody had run.

---

## Note added at it.20: two paths in this record have moved

`scale/window_sweep.py` and `scale/sign_dependence_probe.py` were retired to
`attic/` by iteration 4's P0.3 move (commit `228a048`, "Follow the shadower
through the attic move"). Both files still exist, at `attic/scale/...`.

**The paths are left as written.** They were correct when this record was made,
and rewriting them would make the record claim knowledge of a move that had not
happened yet. A reader following either path should look under `attic/` — which is
what `tests/loop/`'s guards do, via a `resolve()` that checks the original location
and the attic before reporting a file missing.

Verified at it.20 by scanning all thirteen R10 records for file references: 87
checked, 85 resolve directly, and these two resolve under `attic/`.
