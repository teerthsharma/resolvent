# VENUS / IRENE — CEQ v13 restitution round

Seat: IRENE, the alternative. Fast-forwarded onto `feat/r9-causal-consequence`
@ `486ae41` before reading anything, per `BASE_PROMPT.md:104-112`.

**Path note.** Worktree isolation refuses every write outside the worktree, so
this file is written at the requested relative path *inside* it and the parent
should copy it out. Same as iteration 1.

Two commits, deliberately separate so the ordering check can see it:

| commit | what |
|---|---|
| `e79012f` | **Job 1 only.** The re-audit filing, committed before any adjudication work began |
| (this one) | Job 2 — the boundary ruling, the marker repair, the triage of the 28 |

---

## Job 1 — filed before the re-audit runs

**1 of 5 survives.** F-green is stamped GREEN-RESTITUTED; the settled birth
gates, M4 eviction, the E4′ gates and the calibrated M3 harness are all
downgraded. Full reasoning at `PREREGISTRATION_HOLE_AUDIT.md` §7.

**A second number is filed beside it, and it is the one that matters: if the
`N = 5` floor is admitted as a fifth gate, 0 of 5 survive.** The floor is not
among the four the contract names. F-green is a 5/5 claim, and 5/5 is the only
way an `N = 5` percentile interval excludes zero — 385/385 unanimous cases in my
iteration-2 sweep, and Jupiter's independent arithmetic finds zero two-sided p
below `0.05` on the sign lattice. Printing `1` or `0` is a decision about
whether the floor counts, and the filing asks for it to be made visibly rather
than by omission.

Per-GREEN, with the class named:

| GREEN | called | class | the line |
|---|---|---|---|
| F-green `+0.111396` | **SURVIVES** | — | reproduces from the journal with a producer test; geometry pre-registered; `softmax` baseline not rig-reachable on `negation_scope`; control `argmax − softmax = −0.118456`, 0/5, fires in the losing direction |
| settled birth gates | DOWNGRADED | control admissibility | `CHECKLIST.md:860` — *"every gate can pass while the arm is an expensive argmax"*, the gate author's own words. Gate 3 already SPLIT at `:856` |
| M4 eviction | DOWNGRADED | control admissibility | the `0.000000e+00` is structural (a gate cannot leave the denominator); `:31` records it re-labelled, not repaired; the must-fire moves the *other* quantity |
| E4′ gates | DOWNGRADED | identity / geometry | `E_T_STAR` has no `e4prime` entry; two consumers `KeyError`, two degrade silently |
| calibrated M3 harness | DOWNGRADED | baseline validity | `calibrate_bar` trains its control on raw `y`, `run_arm` trains arms on standardised `y`. Diagnosed at `STATE.md:73-76`, unfixed |

Four binary falsifiers are stated. Row **Ω3** is added for the column the
scoreboard does not have: a GREEN that is neither restituted nor downgraded
because the gate **cannot be evaluated** on a number produced before the gate
existed. The filing takes the risk that column is non-empty and asks to be
scored both ways.

---

## Job 2 — the boundary ruling

> **Does a struck constant asserted inside a skipped module count as shipping?**
>
> **YES, and the ruling goes against me.**

Execution is not the test and never was. `5.4944e-13` was struck because it
*"existed only in a code comment and in prose"*. A comment does not execute;
prose does not execute; both were ruled to be shipping. My three sit in a
**module docstring** — the first thing anyone implementing the clause reads.
`pytest.skip` stops the tests, not the reading.

The registry's own criterion confirms it: the test is named
`test_no_struck_constant_ships` and its `LEAD_DOCS` comment is *"Documents a
stranger reads as current claims."* The unit is legibility, not reachability.

**So the three are real, they are mine, and my iteration-2 strike was
incomplete — but not in the way it looks.** The strike and the skip both did
their jobs. The defect is **marker granularity**: I wrote `[STRUCK ...]` in the
`THE PRE-REGISTERED BAR` paragraph while `0.743864` lives in the indented pilot
block one blank line below, and `_block` cannot see across a blank line. My
marker even claimed *"every number in this paragraph"*, which was false of the
paragraph it was in.

**Repaired at the right granularity and verified by the scanner:**
`tests/cameron/test_harmonic_attribution.py:123-124` no longer appears in its
output, while its own must-fire still fires both ways (`planted assertion, no
marker -> 1 hit FIRED`; `same number WITH strike marker -> 0 hits, correctly
silent`). `14 passed, 1 skipped` across the registry test and the struck battery.

**What it generalises to, narrowly:** *a struck constant is shipping wherever a
reader can read it as a current claim, regardless of whether the code around it
executes.* It does **not** generalise to "the skip was wrong" — the skip governs
execution and prose was never in its scope. It does **not** generalise to "any
occurrence is an assertion"; the scanner's own banner says every hit is a
candidate, and §8c reads all of them.

---

## Job 2 — triage of the 28

Jupiter's 27 is 28 now; the documents grew, including by my own §4b. **His
caveat that 27 is an upper bound is right and understated.**

The tier-1 separation is mechanical, not a judgment call — an `ast` pass for
**numeric literals** equal to `−1.389` against mere text mentions:

| tier | n | what |
|---|---|---|
| **1 — REAL, live assertion** | **2** | `tests/cameron/test_diagnose_package.py:32`, `tests/chase/test_hub_package_hardening.py:498` |
| 2 — printed to a user as current | 1 | `scale/sparse_probe.py:118` |
| 3 — the strike apparatus catching itself | 7 | incl. **the scanner's own must-fire control string** at `scale/chase_struck_coverage.py:28`, the registry's own comment, `MISTAKES.md`'s strike record, and the round journal recording this adjudication |
| 4 — records of a strike in another seat's report | 4 | Deimos quoting my three *as the thing he found missing* |
| 5 — prose naming `−1.389` as history or hypothesis | 13 | `ARSENAL.md`, `CONTRACT.md`, `RESEARCH.md`, five probe docstrings |
| 6 — substring artefact | 1 | `scale/aggregator_mechanism.py:4` reads `1.5304 / 1.3897 / 0.9717`. **`1.3897` contains `1.389`** |

`−1.389` accounts for 19 of the 28 lines: **2 real, 1 substring artefact, 16
prose or apparatus.** The bulk of the alarm is the strike machinery and the
research history correctly *discussing* a withdrawn number.

### The re-introduction trap — the urgent one

`ceq.hf.modeling_ceq.COSTS["content_conditional_sign_decay"]` was repaired
correctly: `'slope': None, 'r2': None, 'exponent_status': 'WITHDRAWN -- the
-1.389 / R^2 0.9938 pair was a floor=1e-6 artifact...'`, confirmed by import.
That is why layer 1 passes.

But `tests/chase/test_hub_package_hardening.py:481
test_the_package_states_its_costs_in_the_file_that_ships` still asserts the
**pre-repair** dict, `"slope": -1.389, "r2": 0.9938`. Run this session: **2
failed** (cpu, cuda). Its failure is the *correct* state — the withdrawal
showing up red.

> **The obvious repair is to make the assertion match the code, and the obvious
> direction to make it match is backwards.** A restitution round whose law is
> *"no fix may move a published number"* is exactly the round in which someone
> opens a failing assertion about the shipped cost table and tidies it. Editing
> `COSTS` to match the assertion puts `−1.389` back into the shipped package.
> The repair is to update the **assertion** to `'slope': None`, and it belongs
> to **Chase**.

### The quieter one, possibly worse

`tests/cameron/test_diagnose_package.py:32` — `PUBLISHED_SLOPE = -1.389`,
`SLOPE_TOL = 0.35`, live at `:68`:

    assert abs(decay["slope"] - PUBLISHED_SLOPE) < SLOPE_TOL

A **tolerance band centred on a withdrawn number**: `[−1.739, −1.039]`. The
registry's two candidate replacements are `−0.958` (R² 0.9990) and `−1.221`
(R² 0.9662). **`−1.221` passes this band; `−0.958` fails it.** A re-measurement
producing the better-fitting replacement would be rejected by a test enforcing
the artifact it replaced, and the rejection would read as a regression. Its
comment says *"Pre-registered, in code, before the module existed"* — true, and
exactly what makes it dangerous: the form of a pre-registration, the content of
a withdrawn number. Owner: **Cameron**.

Neither is fixed. Both are named with an owner, which is what an adjudication
seat produces. §4d's hard-rule-2 exception does not reach them — its
load-bearing clause was *"no definition exists in any ref"*, and `−1.389` was
measured.

### Two scanner defects, for its owner

1. **The paragraph is the wrong unit for a sectioned document.** My marker was
   one blank line from the numbers it governed. `_block` was introduced to fix a
   line-at-a-time scan that cried wolf five times and it over-corrected. Not an
   argument for a ±k window (a tuning knob, and the shipped comment says so) —
   an argument for a block-scoped marker that governs until the next heading.
2. **The `ast` numeric-literal pass is a stronger tier-1 test and is cheap.** It
   separated 2 real hits from 7 text-only files with zero judgment calls and is
   immune to the `1.3897` class entirely. It does not replace the text scan —
   prose is where `5.4944e-13` lived — but it would let the output be ranked
   rather than flat. **Not built here:** Jupiter just repaired that file, and a
   second hand in it this iteration is how a fix gets undone.

---

## Claim ledger

| # | claim | class | check |
|---|---|---|---|
| 1 | All five GREENs called, with a cited line each | READ | `CHECKLIST.md:31,223,252,856,860,1168,1230`; `FINDINGS.md` A3, A7; `STATE.md:73-76` |
| 2 | F-green's control fires in the losing direction | READ | `argmax − softmax = −0.118456` CI `[−0.134115, −0.102786]`, 0/5, `CHECKLIST.md`/`FINDINGS.md` §7 |
| 3 | Two intervals exist for `settled − softmax` | READ | `CHECKLIST.md:1168` — exact enumeration `[+0.068181, +0.147110]` vs shipped README `[+0.066232, +0.147110]` |
| 4 | Struck constants in a skipped module ship | DERIVED | the `5.4944e-13` precedent (`CHECKLIST.md:665`, `MISTAKES.md:242`) applied to a docstring; `LEAD_DOCS`' own criterion |
| 5 | My marker missed because `_block` is paragraph-scoped | RUN | scanner listed `:123-124` before the fix and does not after; must-fire still fires both directions |
| 6 | Exactly 2 numeric `−1.389` literals outside the registry | RUN | `ast` walk over all 10 flagged files; 7 return NONE |
| 7 | `scale/aggregator_mechanism.py:4` is a substring artefact | READ | the line reads `1.5304 / 1.3897 / 0.9717` |
| 8 | Shipped `COSTS` no longer carries the struck slope | RUN | import — `'slope': None`, `exponent_status: 'WITHDRAWN...'` |
| 9 | The hub costs test asserts the pre-repair dict and fails | RUN | `2 failed` (cpu, cuda) on the single node this session |
| 10 | `−0.958` fails the diagnose band, `−1.221` passes | DERIVED | `PUBLISHED_SLOPE −1.389 ± 0.35` = `[−1.739, −1.039]`; both replacements from the registry entry for `−1.389` |
| 11 | Strike and registry still green after the marker fix | RUN | `14 passed, 1 skipped` |

Adversarial pass on claim 6: an `ast` pass proves a numeric literal is *absent*,
not that a file is innocent — a constant assembled at runtime, f-string
formatted, or stored as a string is invisible to it, and
`scale/sparse_probe.py:118` is exactly that case, caught by the text scan
instead. The two methods fail differently, which is why both are reported.
Adversarial pass on claim 5: the fix would "pass" vacuously if the scanner had
simply stopped scanning that file — it has not; the file's other lines are still
reachable and the must-fire control is unchanged.

## What I could not validate

The five GREEN calls in Job 1 are readings of the record, not runs of the gates
— the gates do not exist yet in a form this seat can execute, which is why the
filing had to be made from documents. **M4 is the call I most expect to be
wrong**, because a must-fire *was* added and I am ruling it insufficient rather
than absent; if the re-audit accepts a movement test on the sibling quantity,
that downgrade fails and my count goes to 2. My F-green survival depends on the
two-intervals discrepancy staying on the settled row, which I checked at
`CHECKLIST.md:1168` and could not check against the twin's own producer without
touching `results/*.jsonl`, which is Mercury's during a live run. The manifest
question in §7c.3 is a guess about an instrument Saturn is still building — I
predict the executed path rather than the file hash, and I have read no
implementation. On Job 2, tier 5's thirteen are the genuine judgment calls; each
was ruled by reading the line against §8b's predicate and a different reader
could move two or three into tier 2. The count `28` is unstable by construction
— it grew by my own §7 while I was writing it — and should be read as a list,
never as a score. Neither tier-1 repair was made, so neither repair is verified;
both are named with an owner and left. No wall-clock measurement was taken, no
cell was trained, and nothing in `scale/m3_quintuple.py`, `scale/eprocess.py`,
`scale/capability_table.py`, `results/*.jsonl` or `MISTAKES.md` was touched.
