# V20 R15 it.21 — SATURN (instruments)

Branch `v17k-gate0`, repo HEAD `207e7b9`, this box (Windows 11, Git for Windows
bash). Two repairs, both against MARS's it.20 strikes, in the order the round's
severity demands: the destructive defect first.

| # | strike | verdict | node that now catches it |
|---|---|---|---|
| REPAIR 1 | MARS it.20 STRIKE 2 — `$WATCHDOG` signals an unverified pid | **taken, route beaten** | `tests/saturn/test_v20_r15_it21_watchdog_ownership.py` (6 nodes) |
| REPAIR 2 | MARS it.20 STRIKE 1 — `(a)+(b)+(c)` moved together survives everything | **taken, route extended** | `test_every_wings_arm_is_named_by_the_price_row_it_cites`, `test_the_wing_ids_are_pinned_to_their_arms_outside_the_citation_set` |
| FLAG | `C20` index row vs `### CORRECTION 20` body heading | **left flagged, still legible** | `test_the_highest_body_correction_has_an_index_row` |

---

## 1. REPAIR 1 — the watchdog killed on a number, not on evidence

### The defect, as MARS filed it

`scripts/iteration_timer.sh` writes `.claude/iteration.watchdog` in `start`
(`echo $! > "$WATCHDOG"`), reads it in `disarm`, and issued `kill "$wpid"` with
no liveness, ownership or start-time probe. `rm -f "$WATCHDOG"` occurred **once**
in the whole script, inside `disarm`, so a watchdog that ran to completion left a
file naming a dead pid. Dead pid plus OS pid reuse means `stop` sends `SIGTERM`
to a process this repo never started.

It is the it.19 defect with the arms reversed. `$PIDFILE` had no writer and
answered in the **permissive** value — a wrong number. `$WATCHDOG` has a writer
and answers in the **destructive** value — a wrong action. Every other defect
this round has found produces a wrong number; this one produces a signal.

### RED, verbatim, against the unmutated shipped script

The script is copied unmodified into a scratch tree; nothing is mutated.

```
FAILED tests/saturn/test_v20_r15_it21_watchdog_ownership.py::test_stop_does_not_signal_a_pid_the_watchdog_file_merely_names
E   AssertionError: `stop` sent SIGTERM to pid 41158 on the strength of
E   .claude/iteration.watchdog alone. The pid was never this iteration's watchdog;
E   nothing in the file says it is, and OS pid reuse makes the number a stranger.
E   it.19 retired $PIDFILE for reading a process fact it could not source -- this
E   channel acts on the same unsourced fact with a signal.
```

Five of the file's six nodes were RED before the repair; the sixth is the probe
calibration, which is a fact about this box and not about the script.

**MARS's caveat honoured.** Every process signalled in this file is spawned
*through* bash and reports its own shell pid (`sleep N & echo $! > pidfile;
disown`). A `Popen` pid on Git for Windows is a Win32 pid that bash's `kill`
cannot address; a test that measures the pid-namespace gap instead of the defect
passes for the wrong reason, which is the error MARS named in his own filing.

### The route shipped — his trap taken, his stamp moved, one probe added

His route was `trap … EXIT` plus *"corroborate a start-stamp before signalling"*.
The trap is taken unchanged. The stamp is **not** sufficient where he put it: the
watchdog can complete, the pid be recycled, and `stop` be called all inside one
iteration, at which point the stamp still matches and the innocent process still
dies. The stamp is necessary and not sufficient, so it moves and a probe that is
sufficient goes in its place.

1. **`trap "rm -f \"$8\"" EXIT`** in the detached watchdog body. The registration
   dies with the process that owns it, closing the stale-file window for every
   exit the shell can see.

2. **`_owns_watchdog` before any signal.** The pid's *own* record is read, not a
   file this repo wrote:

   ```bash
   _owns_watchdog() {
     local c="/proc/$1/cmdline"
     [[ -r "$c" ]] || return 1
     tr '\0' ' ' < "$c" 2>/dev/null | grep -qF -- "$STATE"
   }
   ```

   The watchdog is a `bash -c` whose argv carries `$STATE`. A recycled pid running
   something else does not carry it, and cannot be made to. No readable procfs, no
   signal — the destructive branch fails **closed**.

3. **The start-stamp, moved from the killer to the killed.** The body now fires
   only if `$STATE`'s `start=` is still the value it was armed against. This is
   what makes (2) affordable: failing closed leaves a stale watchdog alive, and a
   stale watchdog that cannot raise `OVERDUE` against an iteration it was not
   armed for costs a process, not a verdict. Without (3), failing closed would
   trade a wrong kill for a wrong `OVERDUE`.

`/proc/<pid>/cmdline` is present under the MSYS2 procfs on this box and is
calibrated **both ways** in `test_the_ownership_probe_can_read_a_live_process_and_can_return_false`:
the marker must be found in a process started with it and must not be found in
one started without. A probe that cannot return false is not a probe.

### The planted negative — the disarm that still disarms

`_owns_watchdog` returning false for everything passes the RED and silently
retires the disarm. That is `and False` in shell clothing, and the round forbids
it. `test_disarm_still_kills_the_watchdog_it_actually_armed` arms a real watchdog
through the shipped `start 1`, reads the pid the script itself registered, and
demands it be **dead** after `stop`:

```
E   AssertionError: `stop` left watchdog pid <n> running. The ownership probe
E   declines a signal it should have issued: this watchdog IS the one `start`
E   armed, and a surviving watchdog raises OVERDUE against the next iteration.
```

That process is spawned by `start` through bash, so it is a pid Git-Bash can
actually signal — the same correction MARS applied to his own victim.

Two further negatives bind the parts a source read cannot:
`test_the_watchdog_body_clears_its_own_registration_on_exit` runs the **shipped
body text**, lifted verbatim out of the script between `nohup bash -c '` and
`' _ "$((MINUTES*60))"`, with a one-second deadline, and demands both that it
fired and that the registration is gone — the trap cannot be credited for a body
that did nothing. `test_a_watchdog_outliving_its_iteration_is_inert` re-stamps
`$STATE` under a sleeping watchdog and demands no `OVERDUE`.

### Result

```
tests/saturn/test_v20_r15_it21_watchdog_ownership.py    6 passed
tests/mars_v20/test_it20_the_retired_channel_left_a_live_one.py
    test_disarm_kills_a_pid_it_never_verified                     PASSED (was the strike)
    test_the_watchdog_channel_has_a_writer_and_the_retired_one_does_not  PASSED
    test_the_watchdog_file_outlives_the_process_it_names          FAILED
```

That last failure is MARS's own **premise** node, and it fails with the message it
was written to fail with: *"the watchdog body now touches its own registration;
re-derive the strike."* The body was given the comment line `# $8 is $WATCHDOG …`
specifically so his premise breaks on the code rather than on prose, and his
`rm -f "$WATCHDOG"` occurrence count stays at 1 rather than being inflated by this
document's own quotations of it.

### The instrument the room depends on, re-verified after every edit

`bash -n scripts/iteration_timer.sh` clean; `bash scripts/iteration_timer.sh check`
run after each of the four edits and again at filing, exit `0`, output
`iteration 23: 9m32s elapsed, 10m28s left of 20m`. The `check` branch was not
touched.

---

## 2. REPAIR 2 — the third witness, and the ruling on `WING_ARM`

### The correction taken against this office

The it.19 Limits paragraph read *"(a)+(c) together survives"*. MARS is right that
it does not: `_arm_of` reads clause (a), the joint node checks clause (b) against
it, so the two-clause move is caught — **by the joint node, not by the it.19
witness**. The manifest looked defended for a reason that was not the new witness.
What survives is `(a)+(b)+(c)` moved together, and it survived everything the
freeze file shipped. That correction is adopted, not softened.

### RED, verbatim, against the unmutated freeze file

```
FAILED tests/mars_v20/test_it20_two_clauses_is_not_two_witnesses.py::test_the_new_second_witness_does_not_catch_the_three_clause_swap
E   AssertionError: W1 is certified `arm_pl` by clauses (a), (b) and (c) at once,
E   while its clause (d) still cites V17_R4_RETAKE_PRICE.md:194 -- the 15.970
E   SMPRIME price -- and not one shipped node fires.
E   assert []
```

### Node one — clause (d)'s cited line, which the manifest editor does not own

`V17_R4_RETAKE_PRICE.md:194` is ``| `arm_smprime` | **15.970** | 17.839, 14.101 |``
and `:195` is ``| `arm_pl` | **1.614** | 1.622, 1.605 |``. The arm name and the
price sit on **one line**. `test_every_wings_arm_is_named_by_the_price_row_it_cites`
requires the derived arm to be named by the line clause (d) cites.

This is the first witness in the file that is a function of a **cited file's text**
rather than of the manifest's own rows. Clause (c) was a third seat in the same
closed loop precisely because a tampering editor moves rows; the price table is
not a row the editor moves.

### Node two — the ground pin, and the ruling MARS asked for

**`WING_ARM` is re-imported. The distinction from the map struck at it.14 is the
direction it is read in.**

it.14 deleted a hardcoded `wing -> arm` map from `_arm_of` and was right to. Used
as a **source**, a written-down copy of a value clause (a) already determines is a
second place for the manifest to be wrong, and the two diverge quietly.

Re-imported here it is not a source. `_arm_of` still derives; the map is the
**assertion target**. Divergence stops being the quietest thing in the file and
becomes the loudest. The difference that licenses it is what MARS's V-26 mutation
makes visible: every clause of this manifest is a **citation**, and an editor who
moves all four together leaves a document that is internally perfect and describes
the other wing. A pin no clause can move is the only thing that move cannot
satisfy. His sentence is the right one — a map removed for being a second place to
be wrong was also the only second place to be **right**.

The import is function-local, because `test_v20_r15_it14_saturn` imports the freeze
module at load and a module-level import here would be a cycle. It is reached
identically by MARS's harness, which calls the node object directly.

### Both nodes fire on the three-clause swap, and neither is redundant

Under `_mutated_text(("a","b","c"))` with the `FREEZE-SHA256` recomputed over the
moved rows, exactly two shipped nodes fire:

```
FIRES: test_every_wings_arm_is_named_by_the_price_row_it_cites
       W1 derives arm 'arm_pl' ... and prices itself at V17_R4_RETAKE_PRICE.md:194
FIRES: test_the_wing_ids_are_pinned_to_their_arms_outside_the_citation_set
       W1 derives arm 'arm_pl' from its own citations, but W1 IS 'arm_smprime'
```

They answer different questions and the four-clause move separates them. Move
(a), (b), (c) **and (d)** together and the price witness goes green — W1 would
then cite `:195`, which names `arm_pl`, and every clause would agree. Only the pin
still fires. The price witness is the one that survives the pin being wrong; the
pin is the one that survives every clause being consistent.

### Result

```
tests/saturn/test_v20_r15_freeze_manifest.py                        22 passed
tests/mars_v20/test_it20_two_clauses_is_not_two_witnesses.py         5 passed
```

MARS's five nodes include his own calibration `test_the_true_rows_pass_every_shipped_node`,
so the true rows are asserted green against the two new nodes before the
falsification is claimed to be answered.

---

## 3. THE FLAG, CONFIRMED STILL LEGIBLE

Index row `C20` and body `### CORRECTION 20` remain different findings sharing a
number. This was flagged at it.20 rather than realigned, because realigning a
published row in an append-only body is a `C4`-class author amendment. It stays
flagged; nothing in this iteration touched `V20_R15_JOURNAL.md`.

The flag is still legible and its numbers have **moved with the round's appends**,
which is the property being confirmed:

```
FAILED tests/saturn/test_v20_r15_it20_saturn.py::test_the_highest_body_correction_has_an_index_row
E   AssertionError: the CORRECTIONS INDEX stops at C31 while the body runs to
E   CORRECTION 32: a lookup on the round's own anti-staleness instrument, failing
E   silently in the trusting direction
```

At it.20 the same node read the gap at its then-current pair. It now reads `C31`
against `CORRECTION 32`. The instrument tracks the appends rather than naming a
frozen pair, so a reader arriving after this iteration is told about *their* index,
not about it.20's.

---

## Limits

The `/proc/<pid>/cmdline` probe is the MSYS2 procfs on this box and Linux procfs
elsewhere; on a platform with neither, `_owns_watchdog` returns false and `disarm`
never signals. That is deliberate and priced — the start-stamp corroboration makes
a surviving watchdog inert — but it is a *behaviour change* on such a platform,
and it has not been exercised anywhere but here. The `trap` is proven on the
shipped body text run with a one-second deadline, not through `start 1`; a
sixty-second wait to watch a `trap` fire was not spent, so the trap's behaviour
under the real sixty-second path is inferred from the same text, not observed.
A watchdog killed with `SIGKILL` runs no trap, which is why the ownership probe
and not the trap is the load-bearing half. The orphaned `sleep` child of a killed
watchdog survives to its deadline; that is pre-existing and untouched. On the
manifest side, the price witness is defeated by a four-clause move and the pin is
the only node that is not, so wing identity in this round rests on a two-entry map
in `tests/saturn/test_v20_r15_it14_saturn.py:39` — a strictly better position than
it.20's, and not an independent one: nothing corroborates that map itself.
`tests/saturn` reports `9 failed, 169 passed` at filing and `tests/mars_v20`
`39 failed`; none of those failures are in the two files this iteration wrote or
edited, and none were introduced by it, but they are open items and this document
does not clear them. The timer reads `iteration 23` while this round calls the
iteration 21; the discrepancy is in `.claude/ralph-loop.local.md`, is not this
office's to amend, and is recorded here rather than corrected.
