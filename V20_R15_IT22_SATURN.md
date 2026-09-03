# V20 R15 it.22 — SATURN (instruments)

Branch `v17k-gate0`, repo HEAD `207e7b9`, this box (Windows 11, Git for Windows
bash). The clock repair first, because an auditor who cannot date a reading
cannot report one.

| # | owed | verdict | node that now catches it |
|---|---|---|---|
| REPAIR 1 | the cap re-armed under a running auditor, silently | **two holes found, both closed** | `tests/saturn/test_v20_r15_it22_timer_rearm.py` (5 nodes) |
| REPAIR 2 | the pin has nothing behind it (this office's own it.21 limit) | **witnessed, 25 files against 7** | `test_the_pin_is_corroborated_by_the_rounds_own_filed_record`, `test_the_corroboration_can_return_false` |
| CONFIRM | did dropping the index's row count weaken the recipe | **it would have — and MARS already found it was never dropped** | recompute `[RUN]`, below |

---

## 1. REPAIR 1 — THE GUARD QUIT AT THE EXACT MOMENT THE CAP FIRED

### The cause, and it is two causes

**Hole A — the it.6 guard stops guarding when the budget runs out.**
`scripts/iteration_timer.sh start` refused a re-arm only inside `if [[ $_e -lt $_b ]]`
— only while the live iteration was **within** its budget. An OVERDUE iteration
fails that test by definition, so `start` fell straight through the guard and
re-armed. Not with a warning, not with `--force`: with `rc=0` and the ordinary
arming line. The one state in which a fresh clock does the most damage — the room
past cap, an auditor mid-audit holding an `OVERDUE` reading — is the one state the
guard was written not to cover.

**Hole B — a reading cannot be dated by the instrument that produced it, and the
log cannot date it either.** `check` printed elapsed and remaining and nothing that
said *which clock*. Two readings minutes apart are therefore indistinguishable from
one monotonic clock, and the only way to catch a reset was to write down the wall
time of every call and subtract by hand. That is precisely what the INSPECTOR did,
and it is why he alone saw it. The second place he would appeal to could not settle
it either: **every `"agent":"timer"` line in `house-events.jsonl` carried no time**,
so the log records *that* an arm happened and never *when*.

**Which one fired on him is not decidable from the record, and that is Hole B's
whole point.** The event log was measured: over all **51** timer events, exactly
**one** arm is not preceded by a close, and it is the known it.6 double-arm
(`iteration 6 armed` twice) that the guard was built for. Hole A leaves an
unpaired arm, and there is none after it.6 — so on the log's evidence his re-arm
came through a `stop`+`start` pair, which is legitimate at an iteration boundary,
arrives as a matched pair, and **erases `$CLOSED` on its way through** — the one
breadcrumb `check` would have shown a running agent. The round's own tests were
cleared as a route: every timer test in `tests/saturn` and `tests/mars_v20` copies
the script into a throwaway ROOT, so the Inspector re-running green tests did not
touch the room's clock.

The honest statement is that **the instrument could not answer the question asked
of it**, so Hole B is the load-bearing repair: it closes every re-arm route at once
rather than one, including the next one.

### RED, verbatim, against the unmutated shipped script

```
FAILED tests/saturn/test_v20_r15_it22_timer_rearm.py::test_start_refuses_to_rearm_an_iteration_that_is_already_overdue
E   AssertionError: `start` re-armed an OVERDUE iteration and said nothing. The it.6
E   guard refuses only while `_e -lt _b`, so it stops guarding at the exact moment the
E   cap fires -- the room past its budget, an auditor mid-audit holding an OVERDUE
E   reading, and a fresh 20 minutes handed to everyone still inside the old one. A cap
E   that resets is worse than no cap, because the room believes it.
E     rc=0
E     stdout='iteration 27 armed: 20 min cap, deadline 11:45:04Z\n'
E     stderr=''
```

```
FAILED ...::test_two_check_readings_across_a_rearm_cannot_be_read_as_one_clock
E   AssertionError: neither reading names the arm it came from:
E       'iteration 27: 0m1s elapsed, 19m59s left of 20m'
E       'iteration 27: 0m0s elapsed, 20m0s left of 20m'
E     The clock was re-armed between these two calls and both readings are consistent
E     with one monotonic clock.

FAILED ...::test_the_event_log_can_say_when_an_arm_happened
E   AssertionError: 2 of 2 timer events carry no timestamp, e.g. {'t': 'dispatch',
E   'agent': 'timer', 'text': 'iteration 27 armed with a 20 min cap'}. Two arms of the
E   same iteration are indistinguishable in the log from one arm, and their order says
E   nothing about their spacing.
```

3 of 5 RED. The other two are the calibration and were green before and after.

### The route shipped — three edits, none of them a kill

1. **The refusal loses its precondition.** `start` now refuses on the *existence* of
   `$STATE`, and names which case it is refusing: `is still live` or
   `is OVERDUE and NOT CLOSED`. A malformed `$STATE` refuses too, because the elapsed
   arithmetic is now used only for the message. Re-arming an overdue iteration was
   never wrong — closing it silently was; the refusal forces the `stop` that puts
   `iteration N closed at Xm` in the log.

2. **`check` names the arm it is reading:** `armed 11:17:35Z` on both the live and
   the OVERDUE line. A reading is now datable without arithmetic — armed-at plus
   elapsed is the wall time of the call — and any re-arm shows up as a moved stamp
   regardless of the route that caused it.

3. **Every timer event carries `"ts"`**, `log_event` and the detached watchdog's own
   `OVERDUE` finding alike, so the log can corroborate or refute what a reading
   claims.

### The planted negative — the routes that must survive the refusal

Refusing every re-arm passes node 1 and stops the loop dead. That is `and False` in
shell clothing.
`test_the_ordinary_iteration_boundary_still_arms` demands the two routes back:
`stop` then `start` returns `0`, `--force` still overrides an overdue clock, and
`check` still prints `\d+m\d+s elapsed` at `rc=0`. It fails with
`"a fresh iteration was refused after a clean stop"` /
`"`--force` no longer overrides an overdue clock"` /
`"`check` no longer reports elapsed/remaining on a live clock"`.
`test_the_it6_guard_still_refuses_a_live_iteration` holds it.6's own case unchanged.

### Result, and one node this repair moved red

```
tests/saturn/test_v20_r15_it22_timer_rearm.py          5 passed
tests/saturn/test_v20_r15_it21_watchdog_ownership.py   6 passed
tests/saturn/test_v20_r15_it19_pid_channel.py          4 passed
tests/mars_v20/test_it18_timer_liveness_defaults_permissive.py  3 passed
tests/mars_v20/test_it20_the_retired_channel_left_a_live_one.py 2 passed, 1 failed
                                                       20 passed, 1 failed
```

The single failure is `test_the_watchdog_file_outlives_the_process_it_names` —
MARS's own **premise** node, red since it.21 by design and with the message it was
written to fail with. Not introduced here.

**Edit 2 moved two of this office's own it.19 nodes red and they were repaired at
the filter, not at the claim.** `_shape` in `test_v20_r15_it19_pid_channel.py`
existed to normalise the volatile seconds out of two `check` outputs before
comparing verdicts; the arm stamp is volatile by construction — the two sandboxes
are built a fraction of a second apart — so it joins the same filter. Both nodes
still compare the whole verdict text. MARS's it.18 equivalent tests a substring and
was untouched.

`bash -n scripts/iteration_timer.sh` clean and
`bash scripts/iteration_timer.sh check` re-run at exit `0` after **every** edit:
`8m13s`, `8m32s`, `8m49s`, `9m20s`, `10m31s`, `11m56s`, `12m51s`, `13m7s` elapsed —
one monotonic sequence under one arm, `armed 11:17:35Z`, which is the property the
repair exists to make legible.

**And the repair caught a re-arm on its first day, unattended.** The check run at
filing reads:

```
iteration 30: 1m19s elapsed, 18m41s left of 20m, armed 11:31:03Z
```

`iteration 27 / armed 11:17:35Z` at `13m7s`, then `iteration 30 / armed 11:31:03Z`
at `1m19s`. The clock re-armed between this office's last two calls, and the
instrument says so **in the reading itself** — no wall times written down, no
subtraction, no auditor's habit required. That is the INSPECTOR's it.21 position
reproduced live and answered. His reading was a re-arm he could only reconstruct;
this one is a re-arm the instrument reports.

### THE `iteration 27` DRIFT, RULED

Not a fault and not a drift. `$STATE`'s `iteration=` is `sed`'d out of
`.claude/ralph-loop.local.md`, whose `iteration: 27` is the **ralph loop's turn
counter**. The round's `it.22` is a different sequence, maintained by hand in
`V20_R15_JOURNAL.md`. They were never the same number and have diverged since at
least it.19. The timer reports its source faithfully and **labels it with the
round's word**, which is the it.19 defect in miniature — an output that does not
name the quantity it holds. Left as recorded rather than renamed: `iteration N`
appears in the OVERDUE marker, the event log and eleven journal entries, and
renaming a published field mid-round is not this office's authority. **Reading rule
for the room: the timer's number is the loop turn, not the round's iteration.**

---

## 2. REPAIR 2 — THE PIN'S WITNESS

### The limit this office filed against its own repair, at it.21

> Wing identity now rests on a two-entry map that nothing corroborates.

`WING_ARM = {"W1": "arm_smprime", "W3": "arm_pl"}` at
`tests/saturn/test_v20_r15_it14_saturn.py:39`. The ruling on *why* it is a pin and
not a source stands. But a pin no clause can move is still a pin an **editor** can
move, and it was the only statement of wing identity outside the manifest.

### The witness, chosen so it is not a fifth clause

Not another document, not another cited line — **the round's own filed record**.
Every office that has written about a wing has written its id beside its arm, in
prose, in its own report: `V20_R15_IT1_SATURN.md:20` (where the ids were assigned),
`IT2_INSPECTOR.md:78`, `IT2_JUPITER.md:38`, `IT10_VENUS.md:139`,
`IT11_MERCURY.md:175`, `IT13_INSPECTOR.md:36`, `IT20_MARS.md:80`, and eighteen
more. To move the pin an editor must now also rewrite reports authored by MARS,
VENUS, JUPITER, MERCURY and the INSPECTOR — documents this office does not own,
which the JOURNAL quotes, and which the round's append-only law forbids amending.
**That is a change of class, not another seat in the same loop.**

`_corpus_votes()` takes one vote per **file** over `V20_R15_IT*.md`, excluding this
office's own it.22 report, and pairs each wing id with the **nearest** arm name on
its line so a row naming both wings votes for both correctly. The node requires the
pin's arm to win, by `>= 5` files and `>= 2x` the runner-up, from `>= 3` distinct
offices — so no single office, this one included, can carry the verdict.

```
W1  arm_smprime 25 files   arm_pl       7      offices: SATURN INSPECTOR VENUS MERCURY MARS JUPITER
W3  arm_pl      23 files   arm_smprime  7
W2  arm_phase    7 files                       (the struck wing, corroborated for free)
```

### The RED, and what it does and does not establish

The full V-26 attack simulated in-process: the pin swapped **and** `_arm_of` made to
derive the swapped arm, which is what a four-clause move produces — every clause
agreeing with every other clause and with the pin.

```
FIRES test_the_pin_is_corroborated_by_the_rounds_own_filed_record
      the pin holds W1 = 'arm_pl' and the round's own record says 'arm_smprime' in
      25 files against 7. The pin is the only statement of wing identity outside the
      manifest and it disagrees with every office that has written about this wing.
FIRES test_the_corroboration_can_return_false
      W1: 7 files say 'arm_pl' and {'arm_smprime': 25} say otherwise.
```

Three shipped nodes also fired in that run — `test_every_clause_b_line_journals_that_wings_own_arm`,
`test_every_wings_arm_is_corroborated_outside_clause_a`,
`test_every_wings_arm_is_named_by_the_price_row_it_cites` — **and they fired for a
reason the simulation created**: the manifest **text** was left true while `_arm_of`
was patched, so those three compared a derived arm against unmoved citations. Under
the real four-clause move the text moves with it and it.21 measured them green. The
simulation therefore proves the new node fires on the attack; it does **not** prove
the shipped set is silent, and that claim rests on it.21's measurement rather than
on this run.

### The planted negative — a probe that cannot return false is not a probe

The corpus is **not** unanimous, and that is the load-bearing fact.
`V20_R15_IT18_INSPECTOR.md:347` and `IT18_MARS.md:165,169` print
`arm_of W1 = arm_pl` verbatim — mutation transcripts — so the **swapped** pairing is
physically present in the round's record, seven files' worth per wing.
`test_the_corroboration_can_return_false` asserts that contrary evidence exists
before it credits the agreement: a method that has never been asked to choose agrees
with the pin because it cannot disagree with anything.

### Result

```
tests/saturn/test_v20_r15_freeze_manifest.py    24 passed in 0.64s   (22 -> 24)
```

---

## 3. THE CONFIRMATION — THE ROW COUNT SHOULD NOT HAVE BEEN DROPPED, AND WAS NOT

**Asked:** did replacing *"over the 19 `C`-rows"* and *"over 31 rows"* with
*"over the `C`-rows"* weaken the recipe. **Answer: it would have, and the round has
already overtaken the question.**

`C33` records MARS's it.22 finding that **neither replacement was made** — the
literal read `over the **31** `C`-rows`, the replace never matched, and the index
published `31` against a recipe returning `32`. A claim to have removed a stale
number, itself stale, inside the entry that named stale counts as the defect.

**The verdict on the merits is that the count belongs there.** A count is the only
part of the recipe that can detect a **deletion**: the digest moves when any row
changes, but a reader recomputing it has no way to know a row was removed rather
than never written. Dropping the count would have traded the one check the digest
cannot perform for immunity to a defect that only ever produces a *visible*
mismatch. The correct repair is the one now standing — the count kept, corrected,
and given **contiguity** and a **range** beside it:

> **INDEX-SHA256 = `24328026eafbf835458a1e53802fd4a0c88a274b12147f72c119d6d59596ec44`**
> over the **33** `C`-rows `C1`..`C33`, **contiguous**, sorted as written.

Recomputed here with the index's own published command, plus the contiguity the
sentence now asserts:

```
rows 33   contiguous True
24328026eafbf835458a1e53802fd4a0c88a274b12147f72c119d6d59596ec44     [RUN]
```

Declared and recomputed **agree**, the row count is **33** as declared, and
`C1`..`C33` are **contiguous** as declared. `test_the_highest_body_correction_has_an_index_row`
is not re-asserted here; it was green at the coordinator's run and this office
appended nothing to the journal.

---

## Limits

Hole A is closed by test and Hole B by test, but **which of them produced the
INSPECTOR's reading at it.21 is not established** — the event log carries no times,
which is the defect Hole B repairs, so the repair postdates the evidence it would
have needed. The log's pairing analysis (51 timer events, one unpaired arm, the
known it.6 one) points at a `stop`+`start` pair rather than the overdue
fall-through, and that is an inference from absence, not a reading. The arm stamp
is second-resolution; two arms inside one second are indistinguishable in it, which
the sandbox node works around with a `1.1s` sleep rather than by fixing. `"ts"` on
the detached watchdog's `OVERDUE` finding is a format-string change on a line no
node exercises — the sixty seconds to watch a real watchdog fire were not spent,
the same debt it.21 recorded against the `trap`. The `_shape` widening in
`test_v20_r15_it19_pid_channel.py` is this office editing its own node so its own
repair passes; the claim is unchanged and the filter's stated purpose covers it,
but it is the shape of change that should be looked at, and MARS is invited to.
On the manifest side, the corroboration is a **nearest-token-on-a-line heuristic
over prose**, not a parse: its ceiling is a sentence naming one wing with the other
wing's arm closer to it, and the margins (25/7, 23/7) are what price that ceiling.
It is also **evidence of consensus, not of truth** — twenty-five files agreeing
would carry a shared error just as faithfully; what it defeats is an *editor*, which
is the threat the pin was built against, and not a founding mistake at it.1. The
corpus grows every iteration and the node's `>= 5` and `>= 2x` floors are fixed, so
a wing added late will fail the floor before it fails the pairing. The four-clause
move plus a pin edit was simulated by patching `_arm_of`, not by mutating the
manifest text; the claim that no shipped node fires under it is it.21's measurement,
not this iteration's. `tests/saturn` and `tests/mars_v20` were not re-run whole at
filing — only the five timer files and the freeze manifest — so the `9 failed,
169 passed` / `39 failed` standing counts from it.21 are neither confirmed nor
cleared here.
