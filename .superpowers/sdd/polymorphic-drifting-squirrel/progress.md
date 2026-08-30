# SDD ledger — plan: C:\Users\seal\.claude\plans\polymorphic-drifting-squirrel.md

Round R9 — causal/consequence attention. Branch `feat/r9-causal-consequence`.
Author away for 15 iterations. Autonomous. Deliverable on return:
`workdoneplanetrum.md`.

## Standing rulings

- `Ruling: parallel implementers permitted, against subagent-driven-development's
  serial default — every planet runs in its own git worktree, so the shared-tree
  conflict that rule guards against cannot occur, and the author explicitly
  required parallel dispatch for all development and research. Cost if wrong:
  merge conflicts at integration, resolvable by replaying a planet's diff.`
- `Ruling: v11.1 Baker Street roles are carried BY the planets rather than
  replacing them — the author confirmed the v11.1 block is his own prompt, and
  also asked for planet-named Opus agents with haiku/sonnet moons. One planet
  holds exactly one role, satisfying "no agent holds two of derive/execute/
  adjudicate". Cost if wrong: renaming, no work lost.`
- `Ruling: work proceeds on feat/r9-causal-consequence, not master. Master
  received the ox-alpha fast-forward as instructed and is now the integration
  target, not the work surface. Cost if wrong: one branch rename.`
- `Ruling: iteration 1 is an audit-and-repair iteration, not a build iteration.
  The author asked to have his old mistakes found and fixed first. No new arm
  or corpus code lands until the mistake taxonomy exists. Cost if wrong: one
  iteration of build velocity, bought back by not building on defects.`

## Phase 0 — complete

- `feat/ox-alpha` fast-forwarded into `master` (8 commits, 0 divergent,
  merge-base == master HEAD). master now at `74e5590`. RUN, verified.
- Branch `feat/r9-causal-consequence` created from master.
- `run_bucket` counting bug confirmed still live on master at
  `scale/bucket.py:149` and `:184-185`. RUN.
- Shared artifacts written: `BASE_PROMPT.md` (evidence rule, vacuity rule, TDD
  contract, three hard rules), `FINDINGS.md` (audit sections A–G with file:line).

## Iteration 1 — six planets dispatched, all Opus, all worktree-isolated

| Planet | Role | Unit | Status |
|---|---|---|---|
| Saturn | WATSON | `MISTAKES.md` taxonomy; kill `impact_hetero` vacuous control (A1); fix `e2_consequence` bar preprocessing mismatch (A7) | running |
| Mercury | LESTRADE | `run_bucket` count (A2); `E_T_STAR` KeyError (A3); dead code (A6); gate placeholder (A5); round pricing table | running |
| Mars | MORIARTY | Hunt unfound mistakes in unaudited surfaces; one attack on the strongest standing GREEN | running |
| Neptune | LINUS | Doc truth B1–B8, incl. the `capability_table` weight-manifest clobber landmine; systems gate on the vector readout | running |
| Jupiter | MYCROFT | M3 Kirchhoff dual oracle; M4 isotonic+Page trend test; M2 RIP line; M1/M5 Welch coherence floor | running |
| Venus | IRENE | Audit five pre-registrations for fall-through holes; file R9 competing prediction before any R9 number | running |

## Deferred / not yet started

- Units 4–8 of the plan (vector consequence label, truncation law, vector
  readout lane, causal pivot selector, sixth arm) — blocked on iteration 1's
  taxonomy and on Neptune's systems-gate cost numbers.
- Units 13, 15, 16 (foreman_looped 4 cells, CUDA 38 cells, Nash revival) —
  execution work, waits on Mercury's pricing table.
- Harvest of the `feat/modest-gates-9099f2` bucket fix — assigned to Mercury.
- Deletion of `feat/modest-gates-9099f2` — hold until Mercury confirms the
  harvest landed. It is the branch's only unique content.

## Iteration 1 — Venus (IRENE) COMPLETE

`Task Venus: complete (merge commit on feat/r9-causal-consequence, DONE_WITH_CONCERNS)`
Delivered `PREREGISTRATION_HOLE_AUDIT.md` (8 fall-through holes, 4 code drifts),
`R9_IRENE_PREDICTION.md`, `venus-report.md`. 1001 insertions.

**Four findings independently re-verified by the controller (RUN, this session):**

- **H-1 CONFIRMED.** `scale/e_ladder.py:279-280` scans `big` over `("e3_t8",
  "e3_t32")` only, then `:288` returns row C whose sentence asserts *every*
  `|delta|` is below `RESOLUTION_13 = 0.027260`. The universal is printed, never
  checked. On the completed CPU ladder `e3_t1` reads `-0.036025`, above the
  floor — the sentence is false on real data.
- **D-1 CONFIRMED.** Row F (`:240-243`) fires on `won("e3_t1") and deep_lost`
  with no `cur["complete"]` guard. F is a theory-death verdict.
- **D-2 CONFIRMED.** Row H (`:267`) fires on `any(lost) and not any(won)` with
  no `complete` guard, and its claim is a universal over rungs.
- Both F and H sit **before** the partial-ladder guard at `:275-278`, so that
  guard never protects them. Row A is correctly guarded at `:252`.

**Live exposure:** the CUDA lane is 22 of 60 cells — partial right now. Reading
it through `verdict()` could print a theory death on incomplete data.

`Ruling: dispatched TITAN (sonnet moon of Saturn) on the verdict() guards
immediately rather than queueing behind Saturn's in-flight work. The defect is
small, isolated to one function, and live against a partial lane. Cost if wrong:
a merge conflict in scale/e_ladder.py against Saturn's branch, resolvable by
replaying one diff.`

`Ruling: Venus correctly labelled her drift findings DERIVED, not RUN — she holds
no execute seat and never ran verdict(). The controller's re-read upgrades H-1,
D-1 and D-2 to READ. Titan's RED tests will upgrade them to RUN. No verdict is
claimed on this until Titan's red output exists.`

### PLAN CORRECTION — unit 6 as written would have produced a diluted null

Venus's structural claim, **re-verified by the controller at
`scale/m3_quintuple.py:302-311`**:

```python
z = x + a @ x
if self.base_cell not in ("softmax", "glance"):
    z = z.clone()
    z[:, s - 1] = x[:, s - 1] + self._alpha(q, k, x).to(z.dtype)
```

The pivot term is written into row `s-1` **only**. Every other row of `z` is
`x + a @ x`, byte-identical across all five arms.

`Ruling: plan unit 6 is amended. Dropping the [:, s-1] index from the readout is
NECESSARY BUT NOT SUFFICIENT — it would yield an [n, s] label on which all arms
are identical at 63 of 64 positions, diluting the contrast rather than sharpening
it, which is exactly the third alternative Venus was offered and the one she
filed. The vector lane must ALSO write the pivot term at every position, not
only s-1. That is a larger change than one index and it makes the arm genuinely
different, which is the point. Cost if wrong: the vector lane measures a null
that looks like a refutation of the whole thesis, and the round is wasted.`

Venus's filed counter-prediction stands against the amended unit 6 as well, and
is to be scored either way when the numbers exist.

## Iteration 1 — Mars (MORIARTY) COMPLETE

`Task Mars: complete (merged, DONE_WITH_CONCERNS). 10 attacks filed, 8 fired,
4 self-refuted before filing. tests/mars/test_mars_r9_iteration1.py 10/10, 51.83 s.`

**Mars corrected the controller first, and was right.** FINDINGS B5 claimed
*"nothing on disk reproduces"* the IMPACT gate numbers. Mars ran all four gates
for the first time and they reproduce the commit body **exactly**:
`1.0039067318` / `0.0673856682` / `0.3243630511` / `0.8793857278` CI
`[0.8300754114, 0.9145472963]`. B5 has been half-retracted in FINDINGS.md — the
"no results file, no test, no CLI" half stands; the numbers were never
fabricated, only never journalled.

`Ruling: FINDINGS.md is a living document and is corrected in place when a planet
falsifies an entry. The controller's audit is not privileged over a planet's
measurement. Cost if wrong: none — a retraction that turns out unnecessary is
cheaper than a false entry five agents are building on.`

### A1 — IMPACT cannot be learned across the split its own harness draws. CONFIRMED.

Re-verified by the controller (READ, this session):
- `scale/m3_quintuple.py:479-480` — eval batch drawn at `seed + 12345`.
- `scale/impact.py:567` — `graph_seed = int(seed % 1000000)`, feeding
  `build_impact_graph`. Train and eval therefore sit on **different graphs**.
- Mars adds: `A` and `B` never enter `x` (channels 2-9 reserved, unused), so
  nothing in the input identifies which graph is present.

Mars's three independent paths (RUN): linear probe reads the train graph at
`4.93e-08` and the eval graph at `1.478` against a mean predictor of `1.000`;
`cos(w_train, w_eval) = -1.08e-08`; label correlation `0.0072`. The same probe
reading `4.93e-08` within a graph also shows the task is **trivially linear**
there — so *"cross-asset propagation is required"* is false as well.

`Ruling: the impact / impact_hetero registration is UNUSABLE as the round's
causal corpus and must not be built on. It is doubly malformed — unlearnable
across its split, trivial within a graph — and would have produced a null that
reads as "the arm cannot do causality" when the defect is in the task. Saturn's
impact_hetero fix (unit 1) stays valuable as a vacuity kill but does NOT
rehabilitate the task. Plan unit 4 builds a fresh vector-valued corpus rather
than adapting impact. Cost if wrong: one corpus built that could have been
adapted — cheap against a round spent measuring a malformed task.`

### The GREEN attack landed, and it is fair

`argmax`'s alpha carries **no `grad_fn`** — measured gate gradient `0.0` against
`52.32` (twin) and `55.15` (settled). Which pivot `argmax` reads is never
trained. So the headline `-0.118456` confounds *mixture vs lookup* with
*trained vs untrained selection*, and `CHECKLIST.md:1206` is not supported by it.
The same reading is voided at `CHECKLIST.md:1207` and credited at `:1206`.

Mars labels this correctly: it shows the gradient path is **absent**, not what a
straight-through argmax would score. The mixture claim is **unsupported, not
refuted**.

`Ruling: the README's "the gain is the mixture, not the equilibrium" claim is
downgraded to UNSUPPORTED pending a straight-through-estimator argmax cell. It is
not withdrawn — Mars refuted the evidence, not the conclusion. Queued as a unit
for iteration 2. Cost if wrong: a published claim stands one iteration longer
than its evidence.`

### Other fired attacks

- **Sign gate is not specific to sign.** Magnitudes redrawn with **zero** sign
  flips still cross the `0.10` bar at `0.1831` CI `[0.1744, 0.2311]` (seed 0) and
  `0.1452` (seed 2); seed 1 reads `0.0701` and does not. 2 of 3. Its own local
  must-fire reads `-0.0011` and is **excluded from the verdict**.
- Gate 4's verdict is a literal string omitted from `all_pass`; its
  `spectral_radius` covariate is `0.88` at every seed **by construction**.
- Decoder FAIL probe is rank 2 of 4 — `deg_q` constant, condition number
  `1.4e15`, contributes `4e-10`.
- The `0 < frac < 1` non-degeneracy guard reads `0.5` even for a label with
  `sd == 0.0` — the guard that exists to catch the fourteenth strike does not
  catch it.
- **`ceq/rips.py`'s "independent derivation" had no referent.** Mars supplied
  one: union-find agrees 6/6, worst dot margin `7.16e-07`. The mujoco #3396 port
  is now validated rather than merely asserted.

**Mars's own stated limits:** A1's "no *arm* can learn it" is DERIVED — a linear
probe was measured, no arm was trained. The sign-gate attack holds on 2 of 3
seeds. Unaudited: `scale/e4_harmonic*`, `scale/eprocess_perdraw.py`, most of
`ceq/nash.py`.

## Iteration 1 — Mercury (LESTRADE) and Saturn (WATSON) COMPLETE

Both merged into `feat/r9-causal-consequence`. `scale/negation_scope.py` and
`scale/impact.py` auto-merged with no conflict. **Integration check RUN by the
controller: 28/28 green** across `tests/mercury/test_r9_mechanical_fixes.py`,
`tests/cameron/test_impact_hetero_is_not_its_own_baseline.py`,
`tests/cameron/test_bar_control_sees_the_arms_preprocessing.py`, 35.12 s.

### Saturn — `MISTAKES.md` delivered, 507 lines, 22 entries

Four classes: 12 vacuous / 7 provenance / 7 measurement / 5 design. Each entry
carries type name, cited instance, design rule. Closes with seven pre-ship checks.
**This was the author's explicit ask and it is done.**

**A1 killed — and NOT by the fix the controller prescribed.** All five consumers
hardcode `heterogeneous=False`, so rebinding the hetero builder alone would have
swapped a vacuous control for a **mislabelled** one (the oracle then reads NRMSE
`1.399930` against its own label). Saturn instead rode the plant bit in the
tensor as `CH_HET` behind one shared `_graph_from_x`. Discriminates **8/8**
draws, both labels non-degenerate, `impact` corpus byte-identical
(sha `5cb97ab298adc015`).

**A7 fixed.** e2 bar `2.446645` BROKEN → `0.699440` CALIBRATED at the shipped
150 steps. Scale-invariance gap `20.826173` → `0.000000`.

`Ruling: Saturn's rewrite of the prescribed A1 fix is ACCEPTED over the plan text.
The plan called for a one-line rebinding; Saturn showed by measurement that it
would produce a mislabelled control, which is a worse defect than the one being
repaired. The spec's binding requirement is a control that discriminates, not a
particular line edit. Cost if wrong: a CH_HET channel occupies encoding space in
a corpus the round has now decided not to use anyway.`

`Ruling: Saturn's edit to the ONE existing test
test_the_consequence_bar_is_broken_at_the_shipped_step_budget is ACCEPTED as a
licit exception to hard rule 2. That test regression-locked the A7 defect, so
repairing the defect necessarily changes it; Saturn preserved both readings via
calibrate_bar(standardise=False) so the defect stays measurable. The rule exists
to stop findings being deleted, and this deletes none. Cost if wrong: one flag
on a calibration helper.`

`Ruling: Saturn's REFUSAL of two dispatch items is upheld. He found no strike
record for the "identical-twin registration" category the controller asked him to
write, and found the "fifteenth vacuous control" ordinal already claimed twice
(DONE.md:1228, PIVOT_EXCLUSION_FALSIFIER.md:50-52), so he claimed no ordinal.
Writing a taxonomy entry the evidence does not support would itself be a
provenance failure in the file whose purpose is preventing them.`

### Mercury — four fixes, measured pricing, and two ladder cells actually run

A2 harvested verbatim from `modest-gates-9099f2` after verifying its premise, and
confirmed live against an 85-record 4-task journal. A3 fixed by **guard**, not by
inventing a dial. A6 raises. A5 `planted_sd` now returned and read by
`pass_nondeg`, shown FALSE under a constant-feature monkeypatch.

**Measured pricing, replacing the stated figures:**
- `foreman_looped` remainder: **2 cells / 415 s / one bucket** — and Mercury ran
  two of the four, so D1 is now **4/6 cells**, not 2/6.
- CUDA lane remainder: **37 cells / 1,901 s** at journal rate, **4,751 s** at this
  box. Contention band 1.75x–2.5x from two independent lanes.
- Cost is **not linear in steps** — 2.77x for 4x, solving to
  **30.7 s fixed + 0.2944 s/step**.
- **`STATE.md:21` over-prices the ladder by 3.0x** (1.68 h measured against ~5 h
  stated); it priced all twelve cells at the `settled` rate.
- **The GPU is SLOWER on the `e3_t2` rung (0.96x)** — the CUDA lane is
  launch-bound, so finishing 37 CUDA cells buys replication, not speed.

### THE DECISIVE FINDING — IMPACT is dead three independent ways

| Planet | Kill |
|---|---|
| Mars | Unlearnable across its own split (train graph at `seed`, eval graph at `seed+12345`, `A`/`B` never in `x`), **and** trivially linear within a graph |
| Mercury | **Does not fit in memory at the size it requires.** `IMPACT_MIN_NODES = 1024`; `paired_arm.py:70-73` trains full-batch, so at `n_train=2048, s=1024` one attention activation is **8192 MiB against the card's 8188 MiB total** |
| Saturn | Still registered against its own docstring's condition; the four gates have never run as a battery |

`Ruling: IMPACT is RETIRED as a candidate corpus for this round, on three
independent kills from three planets that did not coordinate. Saturn's CH_HET
repair and Mercury's A5/A6 fixes stay — they make it a correct instance of a task
that has not earned admission, which is the honest state. Plan unit 4 builds a
fresh vector-valued corpus. Cost if wrong: a corpus built that could have been
salvaged by changing the training loop's batching — which is a larger change than
building the corpus.`

### Carried open

- **`E2_STEPS = 600` was chosen under the defect Saturn just removed.** Nobody has
  re-derived the budget `e2_consequence` now needs. Blocks the one rung the theory
  actually predicts on.
- FINDINGS A3 (overstated), A6 (wrong lines), B4 (mostly wrong) corrected in place.
- 11 pre-existing failures in `tests/foreman/test_harmonic_attribution.py`
  (missing `absorbing_boundary_kernel`) and 5 in
  `tests/foreman/test_journal_thread_binding.py` — both confirmed present at
  `74e5590`, untouched by either planet.

## Iteration 1 — Neptune (LINUS) COMPLETE. The systems gate FAILS the lane as planned.

Merged. `tests/neptune/test_capability_table_truth.py` 7/7 (3 RED first) and
`tests/chase/test_capability_table.py` 16/16.

### Gate finding 1 — the plan's unit 6, as originally written, was worthless

Measured across **256 drawn instances**: the vector readout is **bitwise
identical** between `softmax` and each of `twin` / `settled` / `argmax` at
positions `0 .. s-2`. Sixty-three of sixty-four coordinates carry **no cell
information**.

- The settled row's share of the loss falls to `1/s = 1/64 = 1.5625 %`.
- **98.4375 % of the gradient would flow through a term bitwise identical across
  all five cells.**
- Diluting the measured `settled − twin = −0.002959` by `1/64` puts the point
  estimate near `−4.6e-05`, which no bootstrap at five seeds can resolve.
  `capability_table.LIMITS` clause (c) already records that a real gap below
  ~`0.05` NRMSE reads NO DIFFERENCE here.
- Neptune's own verdict: *"It is free precisely because it computes nothing
  new"*, and a control whose output is 98.4 % constant across the arms it must
  separate **is vacuous by the repository's own rule 4**.

**Three independent routes reached this: Venus predicted it from the code, the
controller confirmed it by reading `m3_quintuple.py:308`, Neptune measured it on
256 instances.** The amended plan (write the pivot term at every position) is
the only version worth building, and this is now RUN-class rather than a ruling
on a hunch.

### Gate finding 2 — the honest version costs 34×

| | current | per-row | ratio |
|---|---|---|---|
| total FLOPs, `n = 8192` | `7.4313e+09` | `1.2650e+10` | **`1.702×`** |
| non-base share of the arm | `5.856 %` | `44.695 %` | |
| ten units (settled+twin × 5 seeds) | **`0.87 h`** | **`≈ 29.6 h`** | **`34×`** |

Measured two ways. Path A (RUN): one `_alpha` call forward+backward costs
`2.077496 − 0.984537 = 1.092959 s/step`; sixty-four of them is `69.95 s/step`.
Neptune also warns **the FLOP model under-predicts this arm by 2× and the gap
grows**, so realised slowdown will be worse than `1.702×`, not better.

Memory is **not** the constraint: `+3.94 MiB` (softmax) / `+9.42 MiB` (settled)
at `n=8192`, `≤ 0.39 %`.

`Ruling: the round does NOT commit 29.6 h. It pilots. Neptune asked for an author
decision; the author is away under a standing autopilot instruction, and a pilot
is neither irreversible nor outside the worktree, so this is mine to settle. The
per-row arm gets built and measured at REDUCED geometry first — smaller n_train
and smaller s, since the dilution is 1/s and the per-row cost is s alpha calls,
so shrinking s pays twice. Two units of signal before ten units of commitment is
exactly the discipline E_LADDER_PREREGISTERED_READING.md section 5 already
encodes, and that document also records its own pilot was 2.18x optimistic
(realised sd 0.109199 against piloted 0.050146) — so the pilot's optimism must be
stated when it is used to size anything. Cost if wrong: two pilot units of
compute, ~1.5 h, and the full bill still has to be authorised on the author's
return.`

### Neptune found three MORE errors in the controller's FINDINGS section B

- **B4 was already fixed** by commit `54148e6`. `scale/arm_s.py` was never the
  stale file; `STATE.md:230`, which still called it unfixed, was.
- **B7's correction list has two bad numbers**: `155→156` is really `164`;
  `107→106` is really `116`.
- The stated reason for CUDA lane separation names a gate that runs
  `m3_capability.Arm`, **not** `QuintArm`. The conclusion survives; the mechanism
  does not.

`Ruling: FINDINGS section B is now DEMOTED to unreliable as a whole. It has been
falsified in six places across four planets (B5 by Mars, A3/A6 by Mercury, B4 by
Saturn and again by Neptune, B7 twice by Neptune). It was assembled from Explore
agent output that the controller did not line-verify. No planet may cite section
B without re-reading the cited line first, and that instruction is going into
BASE_PROMPT.md. Cost if wrong: agents spend a few extra reads confirming entries
that were fine.`

### A moon went rogue, and the topology has a failure mode

Neptune's moon read a **pre-fast-forward tree**, classified its own planet's
correction as a **suspicious peer message**, disregarded it, and edited two files
from stale readings. Neptune rejected the moon's entire result and re-measured by
hand.

`Ruling: BASE_PROMPT.md gains a mandatory first action — verify HEAD and
fast-forward before reading anything — plus a moon clause: state the required
HEAD in the moon's own prompt, hand it file CONTENTS rather than paths to
re-read, and re-verify what it returns. Three of six planets (Mars, Saturn,
Neptune) were cut 8 commits behind at ac47049, so this was systematic, not bad
luck, and it is the controller's dispatch defect. Cost if wrong: a few redundant
git commands per agent.`

### Carried open

- **The generated tables still lie.** `capability_table_v0.*`, `v1.*` and
  `ceq/hf_artifact/capability_table_v0.json` retain the `--task` denial in stored
  strings. The generator is fixed; regenerating is a **table cut**, which is a
  measurement decision — assigned to Mercury, who holds execute.

## Iteration 1 — Titan and Jupiter COMPLETE. Iteration 1 closed.

### Titan (moon of Saturn) — the verdict guards are in

`tests/cameron/test_verdict_guards_partial_ladders.py` **3/6 RED before, 6/6
after**; the only other `verdict()` consumer, `tests/chase/test_m3_ladder_task.py`,
held **14/14** on both sides. Controller integration check: **20/20**.

This upgrades H-1, D-1 and D-2 from READ to **RUN**. Rows F and H can no longer
claim a theory death on a partial ladder, and row C can no longer print a
universal it scanned only two rungs for.

`Ruling: house-events.jsonl conflicted on this merge and will conflict on every
parallel planet merge, because tests/cameron/conftest.py:39 appends one JSON line
per test to it on every run. Resolved by union and registered
"house-events*.jsonl merge=union" in .gitattributes so git does it automatically
from now on. Four pre-existing invalid-escape lines were verified present at HEAD
before the merge and left untouched. Cost if wrong: a log file keeps duplicate
rows, which it is already designed to tolerate.`

### Jupiter (MYCROFT) — 2252 lines, `tests/jupiter/` 27/27 green

Delivered `scale/kirchhoff.py`, `scale/page_trend.py`, `scale/rip_line.py`,
`scale/coherence_floor.py`, and `results/e_ladder_trend.txt`.

#### M4 says RISES — and Jupiter printed the three things that undercut it

Page's `L = 139`, exact `p = 0.016724`, permutation `p = 0.016255`. Isotonic top
`+0.016035`, CI `[+0.002826, +0.033167]`. Then, against his own result:

- The **same bootstrap without the monotone constraint** reads
  `[-0.004711, +0.033167]` — **covers zero** — and its lower bound reproduces the
  shipped `ci_lo` exactly. The isotonic constraint is what manufactures the
  positive interval.
- PAVA pooled in **13.07 %** of resamples and lifted the bound `+0.007537`.
  Pooling can only ever raise a low top.
- The trend clause survives **2 of 5** seed deletions.
- **Row G already credits 3 of the 4 rungs nothing** — both cells sit above
  predict-the-mean at `t2`, `t8`, `t32`.

Jupiter's verdict, and it is the right one: **"RISES is an ordering, not a
capability."** All three caveats print beside the verdict rather than in a
footnote.

`Ruling: M4 is ACCEPTED as the adjudicator and its answer on the existing ladder
is recorded as RISES-as-ordering, NOT as a capability claim. The v11.1 amendment
specified M4 to replace eyeballing four rungs, and it did — but it would have
returned a headline the evidence does not support had Jupiter not built the
unconstrained comparison beside it. Any future use of M4 must print the
unconstrained CI next to the isotonic one. Cost if wrong: a verdict reads more
cautious than it needed to.`

Calibration (200 drawn tables per arm): flat truth → size clause alone `0.325`,
full `RISES` `0.040`; rising truth → `0.810`. **The trend clause gates the
miscalibrated one** — the conjunction is calibrated even though a clause is not.

#### THE AUTHOR'S OWN CONSTANT IS WRONG, verified two independent ways

The v11.1 amendment states random role coherence `≈ 0.147` at `d = 256, k = 16`.

Measured: **`0.174795`**, Monte-Carlo CI `[0.174460, 0.175131]`, corroborated
**`0.174499`** by order-statistic quadrature — two paths that fail differently,
agreeing. The author's figure is **15.8 % low**.

**Cause:** the union bound runs over `C(k,2) = 120` pairs, not over `k`.

The Welch floor is exactly `0` at `k ≤ d`, as stated. **M5's conclusion
strengthens rather than weakens**: the scramble control's expected residual is
*larger* than the author believed, so calibrating that control to zero is even
more clearly vacuous than the amendment argued.

#### M3 Kirchhoff dual oracle — clean, with one honest hole

Agreement gaps `2.220446e-16` / `8.992806e-15` / `9.636736e-14` at 9 / 62 / 1202
nodes. **The must-fire fires**: a planted off-by-one moves `ω` by `0.175`–`0.316`
and is caught **6/6**. Law wired into `e4_harmonic.measure()`.

Stated limit: `absorbing_chain` and `fixed_point` remain callable directly,
bypassing the law, and **both oracles share `case_graph` — so a builder defect
fools both.** The dual-oracle law catches solver bugs, not builder bugs. That
distinction is now on the record rather than assumed away.

#### Second moon defect of the round, caught by its planet

Jupiter's M2 moon shipped two beds **both at `n/s = 16`** and reported `C`
"exactly stable" — which was forced by the design, since `ln(n/s)` is a single
number there. A vacuous control. Jupiter caught it and added ratio-varying beds:
`C = 1.44270` at `n/s ∈ {16, 64}`, **`1.80337` at `n/s = 4`**. `C` is per-bed and
reusing it across beds is a guess.

`Ruling: two of the round's moons (Neptune's, Jupiter's) produced work their
planet rejected, and in both cases the planet caught it. The topology is working
as designed — the planet is the review seat — but moon output is now treated as
UNVERIFIED by default and must be re-measured by its planet before entering a
report. That clause is already in BASE_PROMPT.md. Cost if wrong: planets spend
time re-checking moon work that was fine.`

## Iteration 1 — Deimos (moon of Mars) COMPLETE. 8 filed, 8 fired, 3 refuted before filing.

Merged. `tests/deimos/` 7/8 in the integrated tree — see the flip below.

| # | Attack | Class |
|---|---|---|
| 1 | **`Eprocess.value`'s `math.exp(logsumexp)` overflows inside `eprocess_perdraw`'s own designed operating range** — crashes at draw 10135 of a realistic 10240-draw pooled run. The decision was already made at draw 67. **And the must-fire calibration battery never exercises the `Eprocess` class that actually reads live data** | RUN |
| 2 | **`nash_operator`'s live path shares one `safe_tau` across the whole batch** — the exact anti-pattern `negation_scope.py:462-466` names and avoids for the identical construction. Measured to roughly halve mean stance magnitude | RUN |
| 3 | **`qre_stance`'s residual-reporting safety mechanism is dead code** — `return_residual` consumed nowhere outside its own definition, no call site checks `tau > tau*`, and a constructed below-threshold game has multiple real equilibria | RUN |
| 4 | **`tests/cameron/test_harmonic_attribution.py` cites nine functions/constants on `negation_scope` that do not exist anywhere in the tree** — an entire "pre-registered" contract clause with fabricated-looking pilot numbers and **zero possible producer**. 11/11 fresh `AttributeError` | RUN |
| 5 | `e4_harmonic.case_graph`'s `lru_cache` shares a mutable adjacency across calls | dormant |
| 6 | `eprocess.calibrate`'s `max_peak` clamp, same overflow class | dormant |

**Attack 4 explains the 11 pre-existing failures** that Saturn and Mercury both
reported and left untouched. They are not a broken test — they are a test whose
producer was never written, carrying pilot numbers nothing could have produced.
This compounds Jupiter's finding that both Kirchhoff oracles share `case_graph`:
attack 5 is the mutable-cache half of that same hole.

`Ruling: ceq/nash.py is REOPENED as a live candidate on the strength of attack 2.
It was measured once at 5.8198 / 4.2107 / 2.6151 OOD NRMSE with every arm above
predict-the-mean and abandoned, and a batch-shared tau that halves stance
magnitude is a sufficient candidate cause for that failure. Deimos correctly
called it A cause on one seed, not THE cause, so this licenses a re-run and not a
conclusion. This is the napkin note's page-5 game-theory thread, so it is squarely
in scope. Cost if wrong: one re-run of a module already written.`

### Deimos's own instrument was bitten by the class he was hunting

`test_qre_residual_is_reported_but_never_read_anywhere_in_the_repo` passed 8/8 in
his isolated worktree and **fails in the integrated tree**. His walk is
`root.rglob("*.py")` excluding only `.git` and `node_modules` — and the primary
repo now holds **nine agent worktrees under `.claude/worktrees/`**, each a full
copy.

Controller measurement: **12 hits — 10 phantom copies of `ceq/nash.py` inside
`.claude/worktrees/`, 1 real, 1 self.** The finding is untouched; the instrument
over-finds.

`Ruling: this is a NEW mistake type and it goes in MISTAKES.md — "a repo-wide
search whose walk includes nested checkouts, reporting phantom hits and inverting
a true absence claim into a false presence." It is the sign-flipped twin of entry
12 (the search structurally incapable of finding anything, which reported zero and
was used to strike a colleague's evidence). Deimos is resumed to fix his walk, to
sweep for the same defect in other repo-wide walks, and to write the entry;
Saturn owns MISTAKES.md and folds it in. Other authors' directories are reported,
not touched. Cost if wrong: one test scopes its walk more tightly than strictly
needed.`

`Ruling: the nine agent worktrees under .claude/worktrees/ are NOT pruned. Their
commits are merged and pruning would be tidier, but the harness owns those
directories, two planets are still live inside them, and a test that breaks
because of what is nested in the tree is the test's defect to fix, not a reason to
delete working directories. Cost if wrong: the tree stays larger than it needs to
and future repo-wide walks must keep excluding .claude.`
