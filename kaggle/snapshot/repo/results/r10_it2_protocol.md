# R10 P0 iteration 2 — spot-check protocol, PRE-REGISTERED

SATURN. Written and committed BEFORE any verification command was run against any drawn row.
The draw (`scale/spotcheck_draw.py`, SEED=10002, `results/r10_it2_draw.txt`) was fixed first and is
not re-drawn. A protocol written after seeing a result is not a protocol, so this file is the commit
that precedes the running.

The draw is heterogeneous by construction: 10 KEEP = 4 test files + 3 scale modules + 3 root docs;
5 ATTIC = 2 test files (vacuous) + 2 scale modules (orphan) + 1 root doc (orphan). "Must pass" is
undefined for a document until defined, so it is defined here, per row type, before execution.

## 0. What is under test, and what is not

The spot-check tests the **DISPOSITION** cell (KEEP / ATTIC). It does not test prose.

- **DISPOSITION FAILURE** — the row's KEEP is wrong (the thing is dead, unreachable, or its stated
  retention ground does not hold) or its ATTIC is wrong. This counts.
- **DESCRIPTIVE AMENDMENT** — a wrong number or wrong word in a non-disposition cell (a test count, a
  record count, a docstring paraphrase) where the disposition still stands. Reported as an amendment
  with its correction; does **not** enter the binomial.
- Exception, pre-registered: a test-count cell reading N>0 where the measured count is **0** IS a
  disposition failure. A file with zero collected tests is not a live test file.

The binomial is defined over **KEEP failures only**, per the brief's own wording. An ATTIC row found
alive is reported as a finding and an amendment; it does not enter the KEEP count. Stated now so it
cannot be traded in either direction after the results are in.

## 1. Exclusions, fixed now so a citation count cannot be tuned later

For every "is it cited / is it reachable" measurement below, these are NOT readers and are excluded:
`AUDIT.md`, `R10_ITERATION_01.md`, `scale/spotcheck_draw.py`, `results/r10_it2_draw.txt`, this file,
`house-events.jsonl`, `house-events-round1.jsonl`, `house-events-round2.jsonl`, anything created by
this iteration, and everything under `.claude/worktrees/` (the AUDIT.md scope rule, MISTAKES.md V-13).
A census naming a row is not a reader of that row; counting itself is how a sheet certifies itself.

## 2. T-KEEP — a test file classed KEEP

PASSES iff BOTH:

- **(T-a) it collects.** pytest imports the file. A collection error is a FAIL with no exceptions: a
  file that cannot be imported is not an instrument.
- **(T-b) its outcome is green, or red-by-design.** Exit 0 is green. Non-zero is red-by-design only if
  every failure is one of:
  - on the `tests/chase/conftest.py:285` KNOWN_RED ledger, which marks it `xfail(strict=True)` so it
    should not surface as a failure at all — and a strict xfail that XPASSes is itself a FAIL; or
  - a `test_claim_*` function in a P1''-carved-out refutation instrument, whose RED is its deliverable.

  Any other red is a FAIL.

Green and red-by-design are reported as **different** measured values, never merged. Conflating them
is how a spot-check lies in either direction.

Class-cell rule: a class error that would flip the disposition is a FAIL; one that leaves it KEEP is
an amendment.

## 3. S-KEEP — a scale module classed KEEP

PASSES iff BOTH:

- **(S-a) reachable without side effects.** `python -c "import scale.<mod>"` exits 0, and importing
  does not run the experiment — measured as no file under `results/` created or modified by the
  import, and import wall clock under 60 s.
- **(S-b) it does something checkable.** At least one of:
  1. a live tracked python importer outside the exclusion list;
  2. runnable as an entry point (a `__main__` guard with a callable body, or `python -m`);
  3. the journal the sheet names for it exists and holds records, or its readings reproduce from
     `results/*.jsonl` at abs=5e-7 by the sheet's own instrument.

FAILS on an import error, or when no leg of (S-b) holds.

## 4. D-KEEP — a root doc classed KEEP

The two existing instruments **do not apply as written**, and the reason is mechanical, not an excuse:
`tests/cameron/test_mistakes_citations_resolve.py:36` binds `DOC = ROOT / "MISTAKES.md"` at module
scope, and `tests/w11/test_w11_claims_resolve.py:37` binds `README = ROOT / "README.md"` the same way.
Each is a single-document instrument; neither can be pointed at `workdone2.md`. Their **method** is
what applies, and it is reused rather than reinvented: the NUMBERED `file:line` extractor and the
`git ls-files` resolution check from the first, the collected-node-id check from the second.

All three drawn KEEP docs carry the same retention ground on the sheet — self-labelled round archive
or per-round work log, superseded by a successor, retained as the provenance of published readings.
That ground is a conjunction and is checked as one:

- **(D1) the provenance re-measures.** The row's `last-run state` cell (N/M readings reproduce at
  abs=5e-7, or a named journal with a record count) reproduces on re-run. **D1 is load-bearing: if D1
  fails, the KEEP fails**, because provenance of readings is the entire stated reason to retain a
  superseded document.
- **(D2) the supersession resolves.** A tracked successor document exists that names this file, or the
  document self-labels as the archive of a closed round.
- **(D3) no citation drift into a surviving file.** Every `path:line` citation in the doc that points
  at a **tracked** file must be in range — the MISTAKES.md P-6 check. A citation to a file that no
  longer exists is recorded as an amendment, **not** a failure: a historical round log naming a file
  later deleted is an accurate record of its round, and demanding otherwise would forbid archives from
  describing history. Stated now, before measurement, because it is the leg most open to post-hoc
  convenience.

PASSES iff D1 and D2. D3 failures are amendments unless they destroy D1.

## 5. ATTIC — "dead" means the stated reason re-measures, not that the sheet says so

"It is dead because the sheet says so" is circular and is the whole thing this check exists to catch.
Each ATTIC row states a reason; that reason is re-measured from scratch.

- **A-vacuous** (`tests/cameron/test_hankel_mustfire.py`, `tests/chase/test_hf_shipping.py`). The
  stated reason is a three-part conjunction and all three must re-measure true: (v1) no front-door
  call — builds its own tensor or graph instead of drawing through the production batch path;
  (v2) 0 of its 6-significant-digit readings reproduce from any `results/*.jsonl` at abs=5e-7;
  (v3) **not** carved out by P1'' — it defines no `test_claim_*` and reimplements no cited external
  mechanism from that source's defining equations.
- **A-orphan, scale** (`scale/orbit_probe.py`, `scale/sparse_probe.py`). Stated: 0 python importers
  and 0 results artifacts, and for `orbit_probe.py` additionally 0 prose citations. Re-measured with
  the blind spot KILL 3 explicitly did NOT rule out — FINDING 1's **constructed** journal path.
  `scale/bucket.py:45` builds `RESULTS / f"{name}.jsonl"`, so a producer can be live while never
  writing the literal string. The check is therefore: importers, `results/` writes at ANY extension,
  AND resolution of every `Journal(NAME)` or bucket name the module constructs against
  `results/{NAME}.jsonl`.
- **A-orphan, doc** (`titan-report.md`). Stated: no other tracked file names this document or its stem,
  measured over 516 tracked text files. Re-measured under the section-1 exclusions.

An ATTIC row is DEAD iff its stated conjunction re-measures true. Any leg that fails makes the ATTIC
wrong, and per section 0 that is a finding and an amendment, outside the binomial.

## 6. The pre-registered verdict, not renegotiable

Under H0 "the sheet is at least 95% right", P(at least one KEEP fails) = 1 - 0.95^10 = 0.4013.

- **0 or 1 KEEP failure** — the row is AMENDED, the census STANDS.
- **2 or more KEEP failures** — the census is REJECTED at p < 0.09 and it re-runs.

The count is stated plainly whichever way it falls. A second failure is not argued into being one.

## 7. The caveat this verdict carries

A uniform draw of 10 is powered against errors spread across the sheet. Whether it is powered against
CONCENTRATED error — a cluster in one directory, one class, or one presumption — is MARS's parallel
iteration-2 calculation and is **not claimed here**. If he measures this draw underpowered, this
verdict is caveated, not discarded, and iteration 3 re-draws stratified.
