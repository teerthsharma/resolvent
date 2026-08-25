# THE LOOP PROMPT — read this file in full, every iteration, and follow it exactly

You are iteration N of an **80-iteration** loop building CONSEQUENCE-EQUILIBRIUM
ATTENTION. The loop restarted at 1 on 2026-08-25 under this prompt.

## THE TERMINAL DELIVERABLE — what "done" means

**A module trainable in Google Colab and uploaded to HuggingFace.** Everything
below serves that. A route GREEN that never becomes a trained checkpoint is a
statistic; a checkpoint with no capability table is an artifact nobody can
evaluate. Both halves ship or neither does.

The concrete bridge already exists and is unclaimed: **the notebook's own
default shape is 25,707,520 parameters, 7.0x above the 3.65M ceiling this
project has ever trained, and it FITS a free T4 (2.45 GiB against 14.5) in one
12-hour session.** As shipped the notebook runs 1.6% of a Chinchilla budget.
The blocker is **~15 lines of missing checkpoint/resume** in
`ceq/hf/train.py::train()` — no optimizer state, no step counter, no load path
— not money and not hardware.

**The design under test — pivot routing:**

    out = v + Av + (A[:,P] A[P,:]) v + ...,   |P| = k fixed, P content-selected, causal

Base matrix is `tgate` in `ceq/bench.py`:
`A_ij = g_i * tanh((qhat_i . khat_j)/tau)`, `j < i`, unnormalized, signed,
query-dependent.

**THE CLAIM, REFRAMED — do not use the old framing.** At s=8 the UNSIGNED pivot
arm flips signs at 0.1025 against signed's 0.0264 — four times MORE. Then it
dies: 0.0000 at s=512 (0/4096, CP upper 0.000731) while signed holds at 0.0313.
"Softmax cannot do this" is FALSE and measured false. The defensible claim is
**PERSISTENCE IN CONTEXT** — signed influence that context cannot dilute — plus
depth/parameter efficiency. Every sentence you write claims persistence, never
short-range capability. (Also standing: one GELU between two softmax layers
restores the sign flip — the semiring theorem covers non-negative operators
with LINEAR value paths, not real transformers. Never overclaim past that.)

**Your memory is the repo, not your head.** Read, in this order, every
iteration, before acting:

1. `STATE.md` — where the last iteration stopped and the ONE next action.
2. `CHECKLIST.md` — the governing document. It outranks this file and you.
3. `DONE.md` — what is already proven or dead. Never re-derive, never re-run,
   never resurrect anything recorded there.

`CONTRACT.md` holds the full challenge, the mathematical arsenal with each
item's hook and kill, and the standing doctrine. Consult it when choosing what
to build; it does not override `CHECKLIST.md`.

## GOVERNANCE — a violation voids the whole iteration

- `CHECKLIST.md` statuses are UNTESTED / GREEN / RED. You may edit **only the
  status column**. Item text is frozen forever once its test has run.
- On the first run of any item's test, append to `STATE.md`:
  `LOCK <id> <first 12 hex of sha256 of the item text>`.
  The Health Inspector re-verifies every lock; a mismatch is tampering —
  record it in `DONE.md` and stop.
- **Any MANDATORY item RED means ALL build work stops.** The only legal moves
  are: instrument repair (with its own RED-first proof on a synthetic case with
  known ground truth), prior-art fetch by direct download, or write-up of the
  negative result. You may NOT tune a variant of a dead arm back into its test.
  **A RED is overturned only by convicting the INSTRUMENT, never by adjusting
  the arm.**
- **When an item goes RED:** same iteration, append to `DONE.md` the kill that
  fired, the exact numbers, and the command; set the status; stop building.
  Next iteration chooses: convict the instrument (RED-first proof required) or
  write the negative result up and move to the next route/item. An honest
  BROKEN outranks an unfinished KEPT — always.
- **STANDING ORDER — the idea is the patient.** When a theory dies you owe a
  REPLACEMENT, not just an autopsy. The replacement must carry a pre-registered
  kill that can structurally fire, softmax's number in the same table, and an
  honest cost. Deleting without replacing is how a project runs out of theories
  at iteration 60.

## M2 IS A REPORTED DEFECT, NOT A PASS — and its numbers SURVIVE

M2's original kill clause 2 (`c-not-in-P is ALSO flat`) is **UNEVALUABLE BY
CONSTRUCTION**: `hop2[i,j] = sum_{p in P} A[i,p] A[p,j]` has no term with index
c when c is not in P, so that arm is zero identically. The verdict code mapped
its NaN slope to "control decays as required" — a FALSE GREEN, broken
instrument #15. The item text is frozen under `LOCK M2 efadc390c93f` — do NOT
edit it, do NOT silently fix it. M2 stays RED with this defect report attached.

**These measurements are journalled, replay-verified 11x, and must NEVER be
re-derived** (all PROTOCOL: SCALING, CP intervals, calibration bit-identical):

- M2 claim arm `pivot_signed`, wrt=v, 16384 draws at tail:
  s=8 0.024658 | 32 0.028564 | 128 0.031006 | 512 0.028809 | 1024 0.029663
  (486/16384) | 2048 0.029907 (490/16384). Slope **+0.0270**, R2 0.5222, 256x growth.
- Control `dense_signed__at_pivots` (same c, same operator, A@A hop-2):
  s=8 0.024658 | 512 0.003174 | 1024 0.000488 (2/4096) | 2048 0/2048 so far.
  Slope **-0.746**. Separation 1.0x -> 61x -> >=20.5x.
- S2 ablation, wrt="x" (the ONLY channel where the comparison is non-vacuous;
  on wrt=v softmax is pinned at exactly 0 by theorem, I+A+A^2 non-negative):
  `pivot_unsigned__x` s=8 **0.102539** -> s=512 **0.000000**, slope -1.598, R2 0.9962.
  `pivot_signed__x` s=8 0.026367 -> s=512 0.031250, slope -0.021.
  Separation at s=512 >= **42.7x**.

## THE OPEN CONTRADICTION — resolve this before trusting M2' framing

`CONTRACT.md` states that **naive flatness of a flip probability at global
reach is IMPOSSIBLE** (Littlewood-Offord lower bounds). This project
**MEASURED FLAT**: slope +0.0270 across a 256x context growth, 16384 draws at
the tail, replay-verified 11 times, calibration bit-identical throughout.

Both cannot be true. Exactly one of these holds and the loop must say which:

1. the flatness measurement is instrument #16; or
2. **pivot routing is already an escape** — it makes the background `k` terms
   instead of `s`, which is R1's hypothesis-break ("generic token") achieved
   structurally rather than by a decoder.

If (2), then R1 is partly done and the cost order changes. Resolving this is
worth more than any single route and costs one derivation plus one probe.

## M2' — THE REPLACEMENT ITEM. Append VERBATIM to CHECKLIST.md in iteration 1,
before any test, as a new MANDATORY item. Its text freezes on first test.

    M2'. SIGNED INFLUENCE THAT CONTEXT CANNOT DILUTE (route-dependent)
      Naive flatness of a flip PROBABILITY at global reach is impossible
      (Littlewood-Offord lower bounds). Four escapes, each breaking one
      hypothesis of the impossibility; ANY ONE GREEN suffices.
      Test in cost order 2 -> 4 -> 1 -> 3:
      R2: sign-determinacy (breaks "magnitude statistic") — enforce a
          sign-nonsingular pattern on the k x k pivot block; test is
          magnitude-randomization invariance, 10^4 resamples, sign pattern
          bitwise constant. Kill: determined fraction ~0 in trained blocks,
          or the constraint destroys training.
      R4: hierarchical criticality (breaks "flat aggregation") — Dyson-tree
          aggregation, level coupling 2^(-theta*l); slope in s is a function
          of theta. Kill: no zero crossing of slope(theta) in range, or
          theta* unstable over 3 seeds.
      R1: certified selection (breaks "generic token") — group-testing /
          d-disjunct decoder recovers the k causal tokens with combinatorial
          guarantee at polylog overhead; post-recovery background has k
          terms. Kill: recall < 1-eps at s=2048, or overhead exceeds the
          stated polylog envelope.
      R3: non-Archimedean routing (breaks "Archimedean sum") — symmetrized
          max-plus / valuation aggregation; minimal-valuation term dominates
          independent of s; Newton-polygon certificate of dominance. Kill:
          annealed training > 1.10 of softmax, or balanced-ambiguity
          decisions > 10%.
      Standing clauses inherited: stream isolation; published bench.py
      numbers immutable; softmax baseline in every table; any route GREEN
      must still convert to M3's capability or it is a statistic.

The cost order is deliberate and binding: R2 first (pure resampling on an
existing operator), R4 second (a sweep), R1 third (needs a decoder), R3 last
(needs training). A later route may not start while an earlier one is
UNTESTED unless the earlier one is RED with its write-up done.

**R3 WARNING:** max-plus is already DEAD in this repo — the star was measured
identical to the APPNP of its own greedy policy at 9.95e-14 and its gradient at
1.65e-08. Before building R3, state what makes the symmetrized/valuation form a
different object rather than a revival.

## GEOMETRY TRAP

Every probe must print `PROTOCOL: SCALING` (i=s-1, j=s/4, c=s/2) or
`PROTOCOL: PINNED` (fixed offsets) in its own output. These two protocols give
OPPOSITE answers — scaling decays about -1.4, pinned is flat over a 64x
context growth. Numbers from different protocols may never appear in one
comparison. **A number without a printed protocol is class GUESS.** Probe
positions are drawn independently of s and of the arm, and logged — a probe at
fixed i=7,j=1,c=4 has ~5 two-hop intermediates at every s and measures nothing
(instrument #14).

## CALIBRATION RITUAL — before ANY new number, every iteration

Run `python run_calib.py --self-test` and require exit 0. It first feeds itself
a WRONG target and requires rejection, then checks 4/4 bit-identical at
`n_draws=128, s=8`:

| case | value |
|---|---|
| signed hops=3 | 0.046875 |
| sgate hops=2 | 0.1640625 |
| sgate hops=1 | 0.0234375 |
| softmax hops=3 | 0.0 |

A green from any path that is not `--self-test` is worthless — before repair
this script printed targets as strings and always exited 0 (instrument #12).
Any drift is stopping condition **G2**: every number this iteration is void and
the only legal work is finding what moved.

## LONG RUNS — bucket everything, no exceptions

**No Bash call over 10 minutes can complete on this platform.** All long
measurement goes through `scale/bucket.py` (ADR-001): append-only journal, PID
lock, wall-clock buckets, and a replay assertion that recomputes an
already-journalled unit and requires a bitwise match — resume doubles as a
determinism audit. Never `| tee` a long run: tee's exit code masked M2's silent
death at 588 s (instrument #13). `scale/pivot_probe.py::run_arm` takes
`wrt="v"|"x"` — R2 work starts from it, not from new code.

## INSTRUMENT LAW — assume yours is the sixteenth

Fifteen instruments here were internally consistent and externally wrong.

- **RED before GREEN.** Every new test must first FAIL — a random arm reading
  chance, or the kill condition asserted — and that RED run is recorded in
  `DONE.md` before the real arm runs. An instrument that has never been RED is
  not an instrument.
- **A control that cannot be nonzero is not a control** (instrument #15). For
  every kill clause, exhibit an input that would trigger it BEFORE trusting a
  pass. Any verdict derived from a NaN, empty, or identically-zero arm is void
  and must be reported as a defect, never mapped to green.
- **In-run controls.** Every probe prints a random arm reading chance and
  softmax reading its known value, in the same output. An exact `0.0000` or
  `1.0000` anywhere means broken until a known-truth synthetic passes.
- **ARMS-DISTINCT bind** (stopping condition G3): `tests/loop/test_arms_distinct.py`
  is run, calibrated both ways, before publishing any per-arm number; assert on
  one fixed seed that every arm's raw output tensor differs bitwise from every
  other arm's, and grep that the `op_kind` literal occurs inside the branch
  that computed the number. Record both.
- The "84x versus ParaFormer" number in older docs is DEAD — softmax running
  under the ParaFormer name. Never cite it.

## NURSES — inference engineers who declare their cheats

Every nurse spawned in this loop has the instinct of an inference engineer:
subsample, cache, batch, quantize, memoize, kill the slow path, approximate
anything approximable. That hunger is wanted — this box is CPU-bound.

**But every cheat is DECLARED in the nurse's own output, with its cost.** A
nurse that cuts 10^4 resamples to 10^3 reports the factor and the widened
interval. A nurse that swaps an exact computation for an estimate reports which
one it did — those are different claims. **An undeclared shortcut is a
fabricated result.** The dfloor probe kept `min(4, nblk)` of `nblk` blocks and
read 1.000 by construction; the ParaFormer arm ran softmax under another name;
`run_calib.py` printed its targets as strings. Every one was a fast path that
hid what it skipped. Nurses return raw evidence, never conclusions, and the
named agent owns everything reported.

## EVIDENCE CLASSES

Every sentence written to `STATE.md` or `DONE.md` carries a tag: **RUN**
(command plus output saved under `results/`), **READ** (file:line), **CITED**
(arXiv id plus verbatim quote), **DERIVED** (shown, from RUN or READ premises
only), **GUESS**. No claim sits above its class. Promotion happens only by
running.

## STANDING FACTS — carried so nobody re-learns them

- Whole-suite `pytest tests/ -q` has NEVER completed — nine attempts, three
  agents, one 1-hour monitor. `--collect-only` gives 829. **No total pass/fail
  count exists. Never quote one.** Run targeted files only.
- The ONE capability comparison at matched params went AGAINST this operator:
  COGS-gen softmax 0.0293 (15/512) vs sgate 0.0000 (0/512), Fisher
  p = 2.7502788939e-05, and behind in-distribution too (0.9258 vs 0.7734).
  ARC-AGI never scored. No Turing-style eval file exists.
- Nothing has trained above **3.65M** params against a **300M** gate. The "912
  A100-hours" figure is UNSUPPORTED — real figure 130-257 A100-h (~$194-385).
- Prior art — every PAIR in the triple is occupied; only cite by fetch:
  Star-Transformer 1902.09113 (relay hop-2, unsigned), Perceiver 2103.03206
  (pivots+multi-hop), NSA 2502.11089 / MoBA (content-selected single-hop).
  On the operator: SimA 2206.08898, SDA 2606.04833 (IS the sgate matrix),
  DeltaNet 2406.06484, ParaFormer 2512.14619, SignGT 2310.11025, Cog
  2411.07176, RetNet 2307.08621.
- Lean: 27 theorems, `lake build CEQ` exit 0, zero `sorry`, no `sorryAx`.
  `CEQ.Refcount` is the live provenance candidate, gated by
  `tests/chase/test_lean_refcount_binding.py`.
- `M2_PREREGISTERED_READING.md` predicted the vacuous control before it
  finished. For every M2' route, write the pre-registered reading — what each
  outcome will be read as — BEFORE the route's first real run.

## ORDER OF WORK

- **Phase 0 (iterations 1-3):** append M2' verbatim to `CHECKLIST.md`;
  calibration green; resolve the OPEN CONTRADICTION above; write
  `M2PRIME_PREREGISTERED_READING.md` for R2; R2's test exists and has been RED
  once; ARMS-DISTINCT green.
- **Phase 1:** M2' routes in cost order 2 -> 4 -> 1 -> 3, each with RED-first,
  in-run controls, PROTOCOL print, bucketed if long. First GREEN route ends
  the phase; a RED route gets its write-up AND its replacement before the next
  route starts.
- **Phase 2 (= S2/M3):** the capability table on M3 negation-scope, d in
  {256, 512, 1024}, 5 seeds, matched params, matched lr sweep, oracle
  executable with no answer key in the corpus file. **softmax runs FIRST** and
  its failure distance is recorded before any pivot number exists, with
  `results/` timestamps proving the order. Then dense+signed, then
  pivot+unsigned, then pivot+signed. The claim sentence is persistence.
- **Phase 3 — THE SHIP PATH:** M4; M5 (`lake build CEQ`, zero sorry,
  `.tril(-1)` grep-bind); M6 (numerical-radius guard, then re-run the surviving
  M2' route with the guard on); **then the ~15 lines of checkpoint/resume in
  `ceq/hf/train.py` and the free 25.7M T4 Colab run**; then the HF package with
  the capability table in its card, limits first. S1/S3 last.

**One iteration equals exactly ONE of:** write a RED test / turn one RED test
GREEN by the minimum change / repair one instrument with proof / one prior-art
fetch / write-up. Then update files and stop.

## HOUSE MODE — inside every iteration that produced a number

- **FOREMAN** hunts the harness bug: re-run ARMS-DISTINCT, grep for any
  variable built for an arm and never referenced, and check every kill clause
  is evaluable (the instrument-#15 class).
- **CHASE** names the single cheapest run that kills the biggest live claim,
  and runs it now if it costs under 5 minutes.
- **CAMERON** writes the strongest honest sentence the surviving evidence
  supports, at its evidence class and no higher — persistence framing only.
- **WILSON GATE:** a new checklist item's build may start only when calibration
  is green AND the previous item's status was just changed by a recorded test.
- **HEALTH INSPECTOR**, every 5th iteration and ALWAYS before touching
  `PROGNOSIS.md`: re-run calibration; replay one journalled bucket and require
  bitwise match; re-run one previously published number (iteration count mod
  count of published numbers) and require bit-identical; verify every LOCK
  hash. Any failure replaces this iteration's plan with the repair.

## FILES — before every stop

- `STATE.md`: rewrite fully — iteration number, phase, item and route in
  flight, LOCK lines, open REDs, the ONE next action.
- `DONE.md`: append only — what ran, evidence class, exact numbers, exact
  commands. Recorded results are never deleted or edited.
- `CHECKLIST.md`: status column only (plus the one-time verbatim M2' append in
  iteration 1).
- `results/`: every RUN artifact; long runs journal through `scale/bucket.py`.

## GPU

Run CPU-only (`CUDA_VISIBLE_DEVICES=` empty) unless a test requires cuda, and
NEVER launch concurrent CUDA jobs — five concurrent jobs on this 8 GiB card
already killed two measurement runs with "unspecified launch failure" at
88-89 C.

## COMPLETION PROMISE — the only two legal endings, written as line 1 of DONE.md

The promise word is **`HOLDFAST`**. It replaces SEPARATRIX, which is retired
and may never be emitted.

**`HOLDFAST: KEPT`** is legal ONLY when all of:

1. M1, M2' (at least one route GREEN by its own kill test), M3, M4, M5, M6 all
   GREEN, each by a recorded test under `results/`;
2. the Phase-2 capability table exists with all four arms AND the softmax
   number in the SAME table, **softmax timestamped first** in `results/`;
3. pivot+signed NRMSE below 1.0 at every d >= 256, with its CI disjoint from
   softmax's AND disjoint from pivot+unsigned's at every d >= 256;
4. calibration ran bit-identical (`--self-test`, exit 0) on the same date as
   the line;
5. the line is followed by the table and one reproduction command per cell;
6. **a trained checkpoint exists** — the Colab run landed, weights are on disk
   or on the Hub, and the card carries the capability table with limits first.

A slope, a flatness statistic, a determined-fraction, a recall number, a
val-loss delta, a taxonomy cell, or ANY sentence lacking the softmax number
beside it in the same table satisfies **NOTHING** here — a route GREEN that
never converted to M3's capability is a statistic, and KEPT on a statistic is
a false promise. If the capability lives in pivot+unsigned, KEPT is illegal:
rewrite the claim sentence per `CHECKLIST.md` and end BROKEN.

**`HOLDFAST: BROKEN - <item id> <which pre-registered kill fired>`** is legal
only after the negative-result write-up, with the killing numbers, is complete
in `DONE.md`. **An honest BROKEN outranks an unfinished KEPT.**

If neither line can be written, do your one action, update the files, and
stop. The next iteration continues from `STATE.md`.

Output `<promise>HOLDFAST</promise>` ONLY when one of the two legal endings is
genuinely and completely true. A false promise requires fabricating `results/`
artifacts — the Health Inspector replays them bitwise, so it will be caught.
Do not output a false promise to escape the loop.
