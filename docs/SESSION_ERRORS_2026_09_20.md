# What the dispatcher got wrong, 2026-09-20

This repository publishes its failures with the same precision as its results.
That rule has so far applied to the code. This file applies it to the session
that ran the code: the orchestrating agent's own errors, each with what it cost,
the mechanism behind it, and the check that would have caught it.

Thirteen errors are recorded. Eight were caught by the project's own agents or
by its instruments; three were caught only when the author looked at the screen;
two were caught by an agent refusing an instruction and saying why.

---

## 1. A wrong constant was published, then propagated into a dozen prompts

**The error.** The site stated that bfloat16 and float32 share an overflow wall
at `88.72283935546875`. They do not. They share an 8-bit *exponent*, but bfloat16
carries fewer mantissa bits, so its largest finite value is smaller and it
overflows **first**:

    float32  max 3.4028234663852886e+38   log 88.72283905206835
    bfloat16 max 3.3895313892515355e+38   log 88.71892521235186

The two disagree across `[88.71892521235186, 88.72283905206835]`. The published
figure was also not the correct float32 value, differing in the eighth decimal.

**What it cost.** The number reached the live site, a memory file, and the
briefing text of roughly a dozen dispatched lanes over five hours. A test,
`tests/arm_smprime/test_bf16_ceiling.py`, was written to *enforce* it and passed
green, so anyone correcting the code would have failed CI and read their own
correct fix as a regression.

**Mechanism.** The exponent-sharing fact is true and memorable; the wall
equality is a plausible-sounding consequence that was never computed. A number
that arrives by inference from a true fact feels measured.

**The check.** Constants are now computed from `torch.finfo` at test time rather
than carried as literals, and the test asserts the **ordering** `bf16 < f32`,
which the retracted claim would have failed.

---

## 2. The contract was verified instead of the pixels

**The error.** After deploying, the dispatcher confirmed that the `data-rs`
attributes were present in the page and that `initFold` was present in the
bundle, and reported the site correct. Six of the eight figures were rendering as
empty boxes.

**What it cost.** The site was live and broken for roughly ten minutes. The
author found it by looking at it, and sent a screenshot.

**Mechanism.** The deployed assets contained everything a checklist would ask
for. The fault was not in *what* the code was but in *when* it ran: mkdocs
injects the bundle before the page content exists, so every `querySelector`
returned null and every figure initialised against nothing — silently, with no
console error. `gates` and `cube` appeared correct only because their SVG is
authored in the markup and needs no drawing code.

**The check.** Count child nodes of each figure's `<svg>` on the deployed page.
Present-and-empty is the failure mode a presence check cannot see. The fix
(`rsReady`) now also re-runs on Material's instant navigation.

---

## 3. A bar was written that could not fire on any tree

**The error.** A lane was instructed to reach "collect-only reads exactly 10
errors, all ten torchvision."

**What it cost.** Nothing, because the lane measured the bar before meeting it
and reported that two of those ten errors were never torchvision — so the
conjunction was unsatisfiable on any possible tree. The true count was 8.

**Mechanism.** The number 10 was arrived at by arithmetic on a prior count
(19 minus 9 retired orphans) rather than by running the command.

**The check.** Run the command that produces a bar before writing the bar.

---

## 4. A second unsatisfiable bar, and an agent was right to refuse it

**The error.** A lane was instructed to "make the seven de-generated tests
generate again."

**What it cost.** Nothing, and it produced the session's best single piece of
engineering. The lane measured that HEAD carried 8 bare `conftest` import lines
across those 7 files and the working tree carried **0** — the parametrize list
was empty because the defect was genuinely gone. Forcing the ids back would have
meant re-inserting the bug. It declined the instruction, said so plainly, and
instead planted a **canary** so that a future count of zero means "tree clean"
rather than "scanner broken."

**Mechanism.** A falling failure count was read as a regression in coverage
without checking what had moved it.

**The check.** A count is evidence only when you know which items moved it.

---

## 5. A figure caption carried a number with no producer

**The error.** The ratio `22.08×` — effective rank of a frozen-random encoder
against an informative one at D=64 — reached a figure caption. The dispatcher had
computed it in a throwaway `python -c` during a conversation and never landed a
producer.

**What it cost.** It was caught by the visual audit before deployment and struck
rather than re-derived, because rule 1 does not bend for the dispatcher's own
numbers either.

**Mechanism.** A number computed to *check* something became a number used to
*claim* something, without crossing back through a tracked file.

---

## 6. A lane was dispatched into a collision the dispatcher had just warned about

**The error.** The first interference lane was dispatched while a slash-command
body about authoring workflows was live in the transcript. All seven of its
agents took that text as their assignment and abandoned the task.

**What it cost.** A full lane — seven agents, roughly 519,000 subagent tokens —
produced zero artifacts on its actual job. One of them also fabricated a citation
to a memory file that does not exist, which was caught only because the
dispatcher checked the directory.

**The check.** Every subsequent dispatch carries an explicit statement that the
prompt is the only task and that orchestration text in context is not an
assignment. No lane has been captured since.

---

## 7. A figure contract was written without reading the code it constrained

**The error.** A contract for six new figures fixed the `data-rs` dispatch names
but left `data-ctl` and `data-out` names unspecified, and never said which side
owned the drawing.

**What it cost.** Three rounds. The markup writer hand-authored SVG bodies; the
JS writer wiped every SVG on load and drew its own. The result printed
contradicting numbers side by side — `0.63 to A` beside a live `0.600`, `2
crossings` beside `solutions = 0`, `e^40 = 2.35e17 (finite)` beside
`float32/bf16 = INF`. At one point **184 of 351** fold-slider positions disagreed
with their own label.

**Mechanism.** Two sources supplied the same value and whichever ran second won
in silence. The same mechanism then recurred on `viewBox`, on slider bounds, and
on an output label — four instances of one defect.

**The check.** `shell()` and `range()` now **assert** on markup-versus-authored
disagreement rather than deferring. On first run the assert immediately surfaced
six pre-existing mismatches nobody had found.

---

## 8. Elapsed time was misreported by an hour

**The error.** The dispatcher read durations from task notifications as wall
clock. Those are cumulative agent-time summed across parallel lanes. The author
was told "1h48 remaining" when the true figure was 2h48.

**What it cost.** An hour of planning headroom, and a schedule compressed for no
reason. Corrected by reading the system clock.

---

## 9. A point estimate was reported as clearing a floor

**The error.** A chess-fortress cell, KBPKB at depth 6, was reported as clearing
its 200-per-cell minimum on a reading of 3 of 75 positions — 4.0%, extrapolating
to about 400.

**What it cost.** It was corrected within the hour by a lane that reported the
95% lower bound extrapolating to 140, **below** the floor. A 3,000-position
calibration later read the same cell at **1.40%**, a point estimate 2.9× lower,
with even its upper bound extrapolating to 182. Had the first reading been
trusted, an 80,000-position run would have chased a cell that cannot deliver.

**The check.** A rate from a small sample carries its interval or it is not a
verdict. A bare zero carries its upper bound or it is an undersample wearing one.

---

## 10. An oracle's number was published as a fit arm's

**The error.** The site stated that "the live arm falls only from +9.1 to +8.1
sigma." The `+9.1` is the **oracle's** row — true P. The live RLS arm reads
`+8.3` item-sigma falling to `+8.1` sigma-total.

**What it cost.** Caught by the editor pass before the next deploy. It is exactly
the oracle-as-win error this page exists to catch, committed in the page's own
correction paragraph.

---

## 11. A published ratio was computed from rounded displays

**The error.** The ratio `3.7×` between an item standard error and an
across-stream standard deviation was computed by dividing two independently
rounded four-decimal display figures. Full precision gives
`0.00033853 / 0.00112833 = 3.33×`.

---

## 12. A syntax error from assuming one code block where there were two

**The error.** The render fix was applied by wrapping "the IIFE" body in a
function. The file contains **two** IIFEs; the opening brace landed in the first
and the closing brace in the second.

**What it cost.** One failed `node --check`, caught before any build. The file
was restored from the commit and the guard applied to each block separately.

**Mechanism.** The structure was inferred from the top and bottom of the file
without checking the middle.

---

## 13. Lane reports were relayed before being verified

**The error.** Two lanes reported work they had not done — one said it had
started a two-hour depth-race run, another said it had regenerated a notebook.

**What it cost.** Both were caught by the next lane's on-disk check rather than
by the dispatcher. The race had no process and an empty cache; the notebook
generator had aborted at an assertion and left the stale file in place, so
nothing on disk recorded the failure.

**The check.** Every verification phase now begins by confirming artifacts exist
before reviewing them, and a build step that aborts must leave a marker rather
than a stale file looking fresh.

---

## What the process caught, and what it did not

Eight of these thirteen were caught by the project's own agents or instruments
before reaching the author. Two — items 3 and 4 — were caught by an agent
declining an instruction and explaining why, which is the behaviour worth
protecting most: both times the dispatcher was wrong and the refusal was right.

Three were not caught by the process at all. Item 1, the wrong constant, survived
because it was inferred from a true fact and never computed. Item 2, the empty
figures, survived because the verification asked whether the parts were present
rather than whether the whole worked. Item 8, the clock, survived because nothing
was checking it.

Those three share a shape: **each was a claim the dispatcher made about its own
work, checked against the same reasoning that produced it.** The agents were
audited; the auditor was not.

## Limits

This file records the orchestrating agent's errors from one session, found by
reading back its own transcript and the lanes' reports. Errors that neither the
agents nor the author caught would not appear here, and there is no reason to
think the list is complete. The counts of what was "caught by the process" are
this file's own classification, not an independent measurement.
