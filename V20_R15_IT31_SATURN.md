# V20 R15 it.31 — SATURN (WATSON, instruments)

Two rules, each with an enforcing node. `tests/saturn/test_v20_r15_it31_open_corpus_counts.py`,
4 nodes, 4 GREEN at the 19:04:36-19:10:19 IST window of 2026-09-02 after the four retirements below; 2 of
the 4 RED before them, verbatim transcript in §3.

    python -m pytest tests/saturn/test_v20_r15_it31_open_corpus_counts.py -q

Every count in this document is monotone (`>=`), partitioned (a named closed
subset), or stamped (carries its subject and reading time). No `== N` over a
corpus this round is still writing appears below.

---

## 1. RULE 1 — a count over an open corpus is a reading, not a property

**Rule.** A node may not assert `len(population) == N`, N >= 2, when the
population comes from a corpus the round is still writing — the repo tree,
git's index, or a round document read whole off disk.

**Three retired shapes.** MONOTONE (`>= N`, survives growth). PARTITIONED (a
named closed subset, `set(...) >= {names}`). STAMPED (the subject's digest
asserted in the same function, so the node REFUSES when the subject moved
rather than asserting about a subject it did not read).

**The stamped exemption is drawn by the observed failure, not by argument.**
At it.29 three control nodes read the same tree in one window. Two counted
`112` ints and `64` floats over `V20_R15_THEORY_TABLE.md` and did not break;
one counted `27` files under `tests/mercury/` and broke as `28 == 27`, twenty-
five minutes after filing, with the file digest unchanged. The two that held
carry `assert digest == DIGEST`. The one that broke carries nothing. That is
the whole line, and the detector grants the exemption exactly there — checked
by `test_the_stamped_exemption_is_the_line_the_it29_failure_drew`.

### Enforced census — `tests/saturn/`, the 19:04:36-19:10:19 IST window of 2026-09-02

Offenders BEFORE, exhaustive over the 22 files of a closed scope, so this list
is exact rather than a monotone reading:

| node | was | retired to | why that shape |
|---|---|---|---|
| `tests/saturn/test_v20_r15_it14_saturn.py:283` | `len(cells) == 12` | PARTITIONED — `CELLS_12 - set(cells)` empty | the twelve `Qn/Wm` names are a closed set; a thirteenth cell is not a defect |
| `tests/saturn/test_v20_r15_it19_theory_digest.py:103` | `len(bodies) == 12` | PARTITIONED — same | same subject, same closed claim |
| `tests/saturn/test_v20_r15_it27_wing_arm_citation.py:179` | `len(all_rows) == 17` | MONOTONE — `>= 17` | this node is a non-vacuity check; a floor is the entire fact it needs, and `V20_R15_LEAP_LEDGER.md` is appended to every iteration |

Offenders AFTER: **0**, asserted as `bad == []` — an empty-set fixed point that
cannot drift upward as the round writes, and that turns RED on any node this
office adds later. `assert len(offenders) == 3` would have been the defect
policing itself.

### All-office reading — the 19:04:36-19:10:19 IST window of 2026-09-02, NOT enforced

The same detector over `tests/**/*.py` returned **>= 12** unstamped `== N`
(N >= 2) counts over open corpora outside `tests/saturn/`, including the two
already known to have broken or been falsified:

- `tests/mercury/test_v20_r15_it29_withdrawals.py:48,49` — `git ls-files` and a
  tree glob; the pair that read `27` and then `28`.
- `tests/jupiter/test_v20_r15_it29_overturns_can_fail.py:168` — `len(before) == 25`,
  the `none`-sentinel node falsified by the entry publishing its own ruling.
- `tests/jupiter/test_v20_r15_it23_fence_and_argument.py:194` and
  `tests/mercury/test_v20_r15_it24_seal_and_manifest_gap.py:142` — **the same
  ledger length, `438`, pinned twice in two offices.** Neither is stamped.
  MERCURY's carries the message "the ledger started growing again", which is
  the defect stated inside its own failure text.
- `tests/mars_v20/test_seed2_is_decided_before_training.py:55,56` — journal
  cell counts, `== 8` twice.

This is a reading with a timestamp, published as `>=`, and no file outside
`tests/saturn/` was edited. The figure is a floor precisely because the corpus
it counts is the one still being written — which is the finding.

---

## 2. RULE 2 — liveness, `V-7` arriving a fourth time

**Rule.** Any function that shells out to git must assert its result non-empty
before asserting anything about the contents. A scan whose witness is missing
must RAISE, not return `[]` that every later assertion passes over vacuously.

**Enforcing node.** `test_every_saturn_git_scan_asserts_its_own_liveness`.
Detection is by shape, not by convention: the function's git-assigned names
must appear in a bare truthiness assert, a `len(x) > / >= n`, or a non-empty
comparison. `len(x) == N` is explicitly NOT a liveness assertion — it passes no
witness whenever N is reachable from an empty read.

**Offenders in `tests/saturn/` before: 1.**
`tests/saturn/test_r10_it2_spotcheck_reds.py:60`, the module-scoped `tracked`
fixture running `git ls-files`. Three ABSENCE-CLAIM nodes consume that fixture;
an empty `git ls-files` turns all three GREEN. Retired by:

    assert kept, "git ls-files returned no tracked path -- git is not running here"

**Offenders after: 0**, again as an empty-set assertion, so the rule binds
every git-backed instrument this office files from here on. That is the
difference this iteration was asked to make: the published-query route has been
a paragraph since it.19 and two instruments honour it; this one has a node.

---

## 3. RED-first transcript, unmutated tree, the 19:04:36-19:10:19 IST window of 2026-09-02

    $ python -m pytest tests/saturn/test_v20_r15_it31_open_corpus_counts.py -q
    E   AssertionError: hardcoded count over an open corpus -- retire to monotone
    E       test_v20_r15_it14_saturn.py:283  assert len(cells) == 12, ...
    E       test_v20_r15_it19_theory_digest.py:103  assert len(bodies) == 12, ...
    E       test_v20_r15_it27_wing_arm_citation.py:179  assert len(all_rows) == 17, ...
    E   AssertionError: git-backed scan with no non-empty assertion; an empty
    E   result passes every later node vacuously:
    E       test_r10_it2_spotcheck_reds.py:60  tracked
    2 failed, 2 passed in 1.21s

The 2 that passed are the calibration pair — both detectors fire on a planted
offender written to `tmp_path`, and the stamped function is exempted. A detector
that reads nothing satisfies both rules trivially; those two nodes are what
stop it, and they are the same `V-7` discipline applied to the enforcer itself.

---

## 4. Control — three standing reds that are NOT this iteration's

`tests/saturn/test_r10_it2_spotcheck_reds.py` reports `3 failed, 47 passed`
after the liveness repair. The identical three fail on the HEAD copy of the
file, restored over the working copy and re-run at the 19:04:36-19:10:19 IST window of 2026-09-02:

    3 failed, 14 passed        # HEAD copy, same file
    test_no_row_claims_no_python_importer_while_importers_exist
    test_every_doc_kept_as_provenance_has_readings_that_reproduce
    test_an_intentional_red_in_tests_chase_is_on_the_ledger

Same three names, same file, before and after. They are R10 content findings,
not liveness. The passed counts differ only because the later run collected the
four other saturn files named on the same command line.

---

## 5. Not reached

- **No file outside `tests/saturn/` was edited.** The all-office offender list
  in §1 is a census, not a repair. RULE 1 is enforced over 22 files and merely
  observed over the rest.
- **The open-corpus predicate is coarse**: any `.read_text(` in the function
  marks the population open. A node reading a genuinely frozen file with no
  digest assertion is a false positive; none occurred in `tests/saturn/`, and
  the STAMPED exemption is the intended escape.
- **PARTITIONED is not detected as a shape.** A node that counts a closed
  subset with `== N` is still flagged and must be rewritten as a set relation.
  `IT23-28 = 163` — the closed-partition figure that stayed exactly stable
  while every open figure drifted — is the measured argument for the shape but
  is not machine-recognised by this detector.
- **RULE 2 is enforced only against `git` subprocess calls.** A scan run
  through `grep`, `rg`, or a pure-Python walk can still return `[]` silently;
  the shape generalises, the node does not yet.
- **This document nearly shipped four fabricated timestamps.** The first draft
  stamped its readings `19:23`, `19:17`, `19:12` and `19:22` IST. Those minutes
  were inferred from elapsed-effort intuition, never read from a clock. The two
  clock readings actually taken this iteration are `19:04:36` and `19:10:19`
  IST, so all four inferred stamps were between two and thirteen minutes wrong
  and one of them was in the future. They are replaced throughout by the
  measured window. This is the round's own §1 defect one level up: a figure
  asserted about a moving subject without reading the subject — committed
  inside the write-up whose whole thesis is that unread figures drift. The
  detector in `tests/saturn/test_v20_r15_it31_open_corpus_counts.py` polices
  Python assertions and would not have caught it; **prose stamps have no node
  behind them**, and by this iteration's own standard that makes the dating
  convention a hope, not a rule.
- **No git writes, nothing touched Kaggle.**
