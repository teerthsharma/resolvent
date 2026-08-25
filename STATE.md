# State

**Updated: 2026-08-25 — ROUND 2, ITERATION 1 complete.**

Governing documents, in precedence order:
1. `CHECKLIST.md` — work-stopping. Outranks everything.
2. `LOOP_PROMPT.md` — the per-iteration law. 80 iterations, promise `HOLDFAST`.
3. `CONTRACT.md` — the challenge, the arsenal with hooks and kills, the doctrine.
4. `DONE.md` — append-only. Round 1 archived at `DONE_ARCHIVE_ROUND1.md` (5,922 lines).

---

## Loop position

| field | value |
|---|---|
| iteration | **46 complete, HOUSE MODE ENDED by user** — max **80** |
| phase | **WORK-STOPPING — M2 RED; repair / prior-art / write-up only** |
| item in flight | **M2 RED, measured on shipped operator (-1.298, 0 flips at s>=128)** |
| LOCK lines | `LOCK M2 efadc390c93f` — re-verified iteration 1 against the archived copy: **byte-identical** |
| calibration | **GREEN** [RUN] `run_calib.py --self-test` exit 0, 4/4 bit-identical |
| promise | `HOLDFAST` — retires SEPARATRIX |

## Open REDs

**M2 - RED, work-stopping IN FORCE.** On the SHIPPED operator (`sgate`) the
pivot arm reads slope **-1.298** against clause 1's **-0.3** bar. The
iteration-1 SUPERSEDED ruling was made on `tgate` numbers and is corrected.
(**-1.826 stood here until iteration 35 struck it** - Cameron's number, which my
own measurement contradicted at iteration 20. The document the loop reads first
disagreed with CHECKLIST.md for fifteen iterations.)

**M5 - RED.** `||A^hops|| = 0.880500` at s=128, hops=2 - nonzero and O(1), so
the exactness hypothesis is unmet. **The 1.471448 printed here since iteration
33 was FABRICATED and is struck** (iteration 35; not reproducible at any of
1,800 settings). Value now pinned by `EXPECTED_TAIL`.

**M2' - RED.** All four routes sign-blind by the twin test.

## THE ONE NEXT ACTION (iteration 46)

**Make the pipeline-exit defect impossible instead of remembered.** Iteration 45
proved the rule does not hold: iteration 40 wrote *"the defect is ANY pipeline"*
into this file, and iteration 45 wrote `| head -3 &&` anyway. **Nine
structure-level defects, and every defence that has actually held is a TEST, not
a sentence in a document.**

The audit commands live in `DONE.md` prose and in ad-hoc shell. The durable form
is a checked-in **Inspector script** that runs the eight checks with no decision
placed after a pipeline, each check asserting a VALUE, and a must-fire control
per check proving it can fail. Then the Inspector pass is a command with an exit
code rather than a sequence of hand-typed shell that can silently invert.

**Blocked on the user, and only this:** the push. `907e3df` on `master`, tree
clean but for the loop's own counter file, `gh` authenticated as **teerthsharma**
with `repo` scope, `teerthsharma/resolvent` free. **One word and it goes.**

### Iteration 45 - Inspector, eight checks clean, and it caught TWO of its own

Clean: calibration; LOCK byte-identical; replay bitwise
(`dense_signed__at_pivots/s2048/b4`); calibration table 4/4 **independently
invoked**; 100 passed on five binds; `lake build CEQ` exit 0 with **0 sorry**;
struck-absence 12 passed; attribution **0 / 0 / 0** per commit.

**Its own instrument #1 - check 4 asserted nothing.** It printed the four target
values out of `run_calib.py` and declared itself done. **That is instrument #12's
exact shape**, the defect where `run_calib.py` *"printed its targets as strings
and always exited 0"*. Re-run by invoking `bench.sign_flip_rate` directly,
bypassing the comparison logic that was the broken part: 4/4 bit-identical.

**Its own instrument #2 - instrument #13, five iterations after I named it.**
`grep ... | head -3 && echo HITS` printed **HITS with no matching lines**:
`head` exits 0 on empty input, so the `&&` fires unconditionally. Re-run
per-commit without a pipeline carrying the decision: **0 / 0 / 0**.

**Iteration 40 wrote, in this file:** *"the defect is ANY pipeline, because the
shell reports only the last stage."* **Then I did it again.**

**NAMING A DEFECT CLASS DOES NOT PREVENT IT.** Iterations 37, 38 and 45 each
reproduced a class recorded one to five iterations earlier. Every defence that
has held here is structural - the calibration gate, the bitwise replay, layer 1
of the struck-constant bind. **The countermeasure has to be a test, never a
rule.**

## Board in flight

Three fellows are running with nurses (inference engineers: cheat hard, declare
every cheat). Their files are appearing — `scale/route_dependency.py`,
`scale/r2_units.py`, `scale/m3_capability.py`. Do not edit those.

- **Foreman** — are R1–R4 four routes or one mechanism in four costumes; which
  kills can structurally fire.
- **Chase** — build and break R2, the cheapest route.
- **Cameron** — the fifth route nobody costed, and the cheapest path to a
  *capability* rather than another statistic.

## Cheap wins queued (from DONE.md, ranked by value per hour)

1. ~15 lines of checkpoint/resume in `ceq/hf/train.py::train()` — unblocks every
   long run including the 300M gate.
2. The free 25.7M T4 Colab run — **7.0× above the 3.65M ceiling**, fits a free
   T4, one 12-hour session.
3. M5 is nearly GREEN — 27 theorems, build exit 0, zero `sorry`; needs the
   grep-bind from theorem hypothesis to shipped tensor.
4. M1 is probably GREEN from existing data — needs the frozen-zero-gradient-cell
   check.

## Machine

Windows, Python 3.11.9, torch 2.5.1+cu121, RTX 4060 Laptop, 8.0 GiB, 24 SMs.
CPU-only unless a test requires cuda. **Never concurrent CUDA jobs** — five
already killed two measurement runs at 88–89 °C. **No Bash call over 10 minutes
completes**; everything long goes through `scale/bucket.py`.
