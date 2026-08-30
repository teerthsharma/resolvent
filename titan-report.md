# TITAN report — D-1, D-2, H-1 partial-ladder guards on `scale/e_ladder.py::verdict()`

**NOTE ON DELIVERY PATH.** The dispatch asked for this report at
`C:\Users\seal\Desktop\New folder (32)\.superpowers\sdd\polymorphic-drifting-squirrel\titan-report.md`.
This agent is worktree-isolated and the Write tool refused that path with
*"Edit the worktree copy of this file instead of the shared-checkout path"* —
there is no worktree copy of `.superpowers` (it sits outside the repo
entirely, as a sibling of the main checkout, not of this worktree). Written
here instead, at the worktree root, so the report is not lost; the
orchestrator will need to copy it to the dispatched path.

Branch: `feat/r9-causal-consequence` (worktree HEAD branch name is
`worktree-agent-a941b534b8e108274`, tracking the same commit chain).
Files touched: `scale/e_ladder.py`, new
`tests/cameron/test_verdict_guards_partial_ladders.py`.

## 0. Housekeeping note

`PREREGISTRATION_HOLE_AUDIT.md` does not exist in this worktree — only in the
main repo root (`C:\Users\seal\Desktop\New folder (32)\PREREGISTRATION_HOLE_AUDIT.md`).
Read from there (READ, not edited) to confirm H-1/D-1/D-2 before touching
anything, since the dispatch's "your worktree already has the audit" did not
hold here. Quoted section text below is `READ` from that path.

## 1. The three defects, confirmed by direct read before any edit

**D-1** (`scale/e_ladder.py:241`, pre-fix): `if won("e3_t1") and deep_lost:
return ("F", ...)` — no `cur["complete"]` guard. Row A's analogous path is
guarded at `:252`.

**D-2** (`scale/e_ladder.py:267`, pre-fix): `if any(lost(t) for t in RUNGS)
and not any(won(t) for t in RUNGS): return ("H", ...)` — no `cur["complete"]`
guard. H's own sentence is a universal ("is positive nowhere") over every
rung.

**H-1** (`scale/e_ladder.py:280-281`, pre-fix): `big = [t for t in ("e3_t8",
"e3_t32") if ...]` feeding the C/D branch — inspects only the two deep rungs,
so a shallow-rung (`e3_t1`/`e3_t2`) delta above `RESOLUTION_13` is invisible
and row C's "every |delta| is below the 13-seed resolution" prints anyway.
Audit note: on the real completed ladder `e3_t1` reads `delta = -0.036025`,
`|delta| = 0.036025 > RESOLUTION_13 = 0.027260` — the trap was one rung's CI
away from firing on real data (it didn't fire because `e3_t2`'s CI excluded
zero and G pre-empted first).

## 2. Test file: `tests/cameron/test_verdict_guards_partial_ladders.py`

Fixture shape verified by `READ` of `scale/e_ladder.py::read()` (`:134-214`)
and cross-checked against the pre-existing fixture helpers in
`tests/chase/test_m3_ladder_task.py::_rung`/`_cur` (`:255-268`), which build
the identical shape. `complete` is derived from the rows
(`all(r["state"]=="RUN" for r in rows)`), never hand-set, so no fixture can
claim a completeness its own ladder doesn't have (vacuity rule #1/#2).

Six tests: three RED-first (one per defect), three must-fire controls (same
row, same result, all four rungs present) — a guard that blocks the row in
every case is a vacuous fix (vacuity rule #4), so each guarded row is also
proven to still fire when genuinely earned.

### RED run, before any fix to `scale/e_ladder.py`

```
$ python -m pytest tests/cameron/test_verdict_guards_partial_ladders.py -q
F.F.F.                                                                   [100%]
================================== FAILURES ===================================
________________ test_row_f_does_not_fire_on_a_partial_ladder _________________
    assert row != "F", f"row F fired on a 2-of-4 ladder: {why!r}"
E   AssertionError: row F fired on a 2-of-4 ladder: 'K-2E FIRES as written at
    LOOP_PROMPT.md:400 -- settled LOSES on a deep-t* rung AND WINS at t*=1.
    Theory death.'
E   assert 'F' != 'F'

________________ test_row_h_does_not_fire_on_a_partial_ladder _________________
    assert row != "H", f"row H fired on a 2-of-4 ladder: {why!r}"
E   AssertionError: row H fired on a 2-of-4 ladder: "SETTLING IS A STRICT
    COST -- the CI excludes zero and is NEGATIVE at e3_t8 and is positive
    nowhere. Not a tie: the iteration measurably hurts at matched
    parameters. K-3E's conclusion holds a fortiori and the twin ships."
E   assert 'H' != 'H'

________________ test_row_c_does_not_print_the_false_universal ________________
    assert row != "C", f"row C printed the false universal: {why!r}"
E   AssertionError: row C printed the false universal: "K-3E FIRES
    (LOOP_PROMPT.md:403) -- E3 is solved EQUALLY by the twin at matched
    parameters on its own home terrain, and every |delta| is below the
    13-seed resolution 0.027260, so 1.8's 'too small to matter at this
    scale' applies. Settling retires; the twin ships."
E   assert 'C' != 'C'

=========================== short test summary info ===========================
FAILED ...test_row_f_does_not_fire_on_a_partial_ladder
FAILED ...test_row_h_does_not_fire_on_a_partial_ladder
FAILED ...test_row_c_does_not_print_the_false_universal
3 failed, 3 passed in 1.27s
```

The 3 passes on that RED run are the three must-fire controls — direct
evidence they are not vacuous: they already hold before any guard is added,
because on a *complete* ladder the new guard's condition is always true, so
they exercise the genuine-firing branch, not the guard.

## 3. The fix — smallest diff that turns all three false

`scale/e_ladder.py`, three edits, all inside `verdict()`:

1. `:241` — `if won("e3_t1") and deep_lost:` → `if cur["complete"] and
   won("e3_t1") and deep_lost:`. On a partial ladder that would have hit F,
   control now falls to the very next check, `if won("e3_t1"): return
   ("E", ...)` — E's sentence ("capacity leakage ... whatever t*=32 says")
   does not depend on completeness and stays true regardless of what the
   missing rungs would have shown, so this fallthrough is not a second
   defect.

2. `:267` — `if any(lost(t) for t in RUNGS) and not any(won(t) for t in
   RUNGS):` → same `cur["complete"] and` prefix. Unlike F, H's natural
   fallthrough (the old `"--"` partial-ladder message) asserted "no rung
   excludes zero" — which would itself be a false sentence whenever a rung
   *was* lost. Fixed by branching the `"--"` message on whether any rung was
   lost: when one was, the message names it and states H's universal
   couldn't be checked at the unrun rungs; when none was (the pre-existing
   fully-empty-ladder case), the original sentence is untouched.

3. `:280-281` — `big = [t for t in ("e3_t8", "e3_t32") if ...]` → `big = [t
   for t in RUNGS if ...]`. D's message is parametrized over `big` in both
   places it appears (the rungs named, and the "Route: 5 -> 13 seeds at {}
   only" prescription), so widening the scan automatically renames which
   rungs D reports — verified by direct invocation (below), not assumed.

```diff
     deep_lost = lost("e3_t8") or lost("e3_t32")
-    if won("e3_t1") and deep_lost:
+    if cur["complete"] and won("e3_t1") and deep_lost:
         return ("F", "K-2E FIRES as written at LOOP_PROMPT.md:400 -- settled "
...
-    if any(lost(t) for t in RUNGS) and not any(won(t) for t in RUNGS):
+    if cur["complete"] and any(lost(t) for t in RUNGS) and not any(
+            won(t) for t in RUNGS):
         lo = [t for t in RUNGS if lost(t)]
         return ("H", "SETTLING IS A STRICT COST -- ...")

-    # every credited rung's CI covers zero
+    # every credited rung's CI covers zero -- or the ladder is partial and H's
+    # "positive nowhere" universal cannot be checked at the missing rungs
     if not cur["complete"]:
+        lo = [t for t in RUNGS if lost(t)]
+        if lo:
+            return ("--", "the CI excludes zero and is NEGATIVE at {}, but "
+                          "the ladder is PARTIAL: row H quantifies over every "
+                          "rung ('positive nowhere') and an unrun rung could "
+                          "still have won. No kill may be claimed."
+                    .format(", ".join(lo)))
         return ("--", "no rung excludes zero, but the ladder is PARTIAL: ...")
-    big = [t for t in ("e3_t8", "e3_t32")
+    big = [t for t in RUNGS
            if t in live and abs(live[t]["delta"]) >= RESOLUTION_13]
```

`RESOLUTION_13 = 0.027260` unchanged — confirmed by `git diff` showing no
touch to that line.

### GREEN run, after the fix

```
$ python -m pytest tests/cameron/test_verdict_guards_partial_ladders.py -q
......                                                                   [100%]
6 passed in 1.23s
```

### Regression check on the pre-existing `verdict()` test file

Not part of the six required tests, run as due diligence since the change
touches shared code with an existing consumer
(`tests/chase/test_m3_ladder_task.py`, which fixture-builds `verdict()` calls
directly, including a row-F case, a row-H case, and the row-C-vs-partial
case at `:325-337`):

```
$ python -m pytest tests/chase/test_m3_ladder_task.py -q
..............                                                           [100%]
14 passed in 4.48s
```

`grep -rn "e_ladder\|EL\.verdict\|verdict(cur"` over the tree found no other
caller of `verdict()` or consumer of its row-message text besides
`scale/e_ladder.py::main()` and this one test file.

### Manual confirmation of message content (RUN)

```python
>>> EL.verdict(cur)   # e3_t1 above RESOLUTION_13, e3_t2/t8/t32 below, complete
('D', "UNDERPOWERED, NOT A KILL -- every CI covers zero, but |delta| at
e3_t1 is at or above the 13-seed resolution 0.027260, so LOOP_PROMPT.md
1.8's 'too small to matter' clause does not apply. Route: 5 -> 13 seeds at
e3_t1 only.")

>>> EL.verdict(cur)   # e3_t1 covers zero, e3_t8 lost, e3_t2/t32 NOT RUN
('--', "the CI excludes zero and is NEGATIVE at e3_t8, but the ladder is
PARTIAL: row H quantifies over every rung ('positive nowhere') and an unrun
rung could still have won. No kill may be claimed.")

>>> EL.verdict(cur)   # all four rungs NOT RUN (empty ladder)
('--', 'no rung excludes zero, but the ladder is PARTIAL: rows A, C and F
quantify over every rung and none of them is available. No kill may be
claimed.')
```

D correctly renamed itself to `e3_t1` (previously invisible to the pre-fix
comprehension); the two `"--"` partial messages are text-distinct and each
individually true; the fully-empty-ladder sentence is byte-identical to
before the fix, preserving
`tests/chase/test_m3_ladder_task.py::test_the_reader_refuses_a_verdict_on_an_empty_ladder`.

## 4. Claim ledger

| # | claim | class | check |
|---|---|---|---|
| 1 | D-1: row F fired with no `complete` guard, pre-fix | READ+RUN | `scale/e_ladder.py:241` before edit; reproduced by `test_row_f_does_not_fire_on_a_partial_ladder` RED run above |
| 2 | D-2: row H fired with no `complete` guard, pre-fix | READ+RUN | `scale/e_ladder.py:267` before edit; reproduced by `test_row_h_does_not_fire_on_a_partial_ladder` RED run above |
| 3 | H-1: row C printed a false universal, pre-fix | READ+RUN | `scale/e_ladder.py:280-281` before edit; reproduced by `test_row_c_does_not_print_the_false_universal` RED run above |
| 4 | The three fixes are not vacuous (F/H/C still fire when genuinely earned, complete ladder) | RUN | 3 must-fire tests pass both before and after the fix (RED run shows `3 failed, 3 passed`; GREEN run shows `6 passed`) |
| 5 | Fix does not regress the pre-existing `verdict()` consumer | RUN | `tests/chase/test_m3_ladder_task.py -q` → `14 passed` after the fix |
| 6 | `RESOLUTION_13` untouched | READ | `git diff -- scale/e_ladder.py` shows no change to that line |
| 7 | No other code path consumes `verdict()` or its row-message strings | READ | `grep -rn` over the tree, results enumerated above |
| 8 | Row D's message auto-corrects to name `e3_t1` once the comprehension widens | RUN | manual `EL.verdict(cur)` invocation, output quoted above |

**Adversarial pass.** For each RED test: would it pass with the guard logic
deleted (i.e. against the untouched file)? No — the RED run above shows all
three failing against the pre-fix file, so each is a real discriminator, not
an algebraic identity. For each must-fire control: does it share an
assumption with its RED sibling that could make both vacuously pass or fail
together? No — the must-fire fixtures set `complete=True` (all four rungs
RUN), the RED fixtures set it `False` (two rungs `NOT RUN`); `_cur()` derives
`complete` from the rows rather than accepting it as a parameter, so the two
fixture families cannot be confused by construction.

## 5. What this did not validate

Did not run `scale/e_ladder.py` against the real completed bucket journal
(`results/m3_quintuple_v2.jsonl`) to see the new row live — that would need
the real journal file and is outside a code-writing moon's remit (`WATSON`
does not execute training or adjudicate results). The audit's own numbers
(`e3_t1 delta = -0.036025`, CI `[-0.118936, +0.062209]`, `e3_t2`'s CI
excluding zero) are `DERIVED` from `PREREGISTRATION_HOLE_AUDIT.md`, not
independently re-read from the journal this session, so the claim "this fix
changes the real ladder's printed row" is not made here — only that the
*code* no longer has a branch capable of printing row C while a shallow rung
sits above the resolution floor, and no longer has a branch capable of
printing rows F or H on a partial ladder. Did not run the full `tests/`
suite (forbidden by the three hard rules) — the two files run here
(`tests/cameron/test_verdict_guards_partial_ladders.py`,
`tests/chase/test_m3_ladder_task.py`) are the only ones touching
`scale.e_ladder`, established by `grep`, but that grep cannot see a caller
that reaches `verdict()` through dynamic dispatch or a string-built import,
if one exists elsewhere in the tree.
