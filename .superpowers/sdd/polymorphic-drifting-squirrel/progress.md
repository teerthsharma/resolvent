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

## Deimos second pass — the walk is fixed structurally

`tests/deimos/` **8/8 green** in the integrated tree. All eight attacks still fire.

Fix: replaced `root.rglob("*.py")` plus a directory blocklist with
**`git ls-files -- '*.py'`** — scoped to the tracked repo rather than the
filesystem subtree, so it is immune to nested worktrees *regardless of how many
exist*. That is the right shape: a blocklist would have needed extending every
time a new nesting appeared.

Swept the tree for the same class (`rglob` / `os.walk` / `Path.glob` /
`glob.glob` / `iterdir`) — **4 other hits, all confirmed safe and left
untouched**: `tests/chase/test_eprocess.py` (scoped to `results/`),
`tests/chase/test_hub_package_hardening.py` (non-recursive, non-matching
pattern), `scale/chase_struck_coverage.py` (already excludes `.claude`),
`tests/chase/test_capability_table.py` (scoped to a local output dir).

Deimos noted honestly that the 12/10/1/1 phantom-hit counts are the controller's,
measured in the integrated tree, and **not independently reproduced by him** —
his worktree contains no nested worktrees. The fix is structural, so it does not
depend on that count being right.

New mistake type written up in `tests/deimos/DEIMOS_REPORT.md` under "Addendum",
for Saturn to fold into `MISTAKES.md`.

## Iteration 2 — in flight

| Planet | Unit | Status |
|---|---|---|
| Saturn | Vector consequence corpus `c1_propagate` — `[n, s]` label, k-hop reading, closed-form truncation law, do()-bit movement test | running |
| Neptune | Per-row arm at **pilot geometry** — write the pivot term at every position, new tuple beside `CELLS`, own `m3_flops` term, degenerate-setting bind | running |
| Mercury | `Eprocess` overflow at draw 10135 of its own 10240 range + point the calibration battery at the class production uses; then the capability table cut | dispatched |

### Still queued, not yet dispatched

- **Straight-through argmax cell** — resolves Mars's GREEN attack, which showed
  `argmax`'s alpha carries no `grad_fn` (gate gradient `0.0` against `52.32`
  twin / `55.15` settled), so the headline `-0.118456` confounds mixture-vs-lookup
  with trained-vs-untrained selection. Blocked: touches `m3_quintuple.py`, held by
  Neptune.
- **`ceq/nash.py` re-run with per-example `tau`** — Deimos's attack 2 reopened it.
  Free file, dispatch when a seat frees.
- **`tests/cameron/test_harmonic_attribution.py`** — nine cited symbols exist
  nowhere; needs a ruling on whether to write the producers or strike the clause.
  Deferred deliberately: it is a pre-registration question, so Venus should see it
  before anyone edits.

## Iteration 2 — Saturn COMPLETE. The vector corpus exists.

Merged. `tests/cameron/test_c1_propagate_registration.py` **23/23** (RED first:
19 failed / 0 passed). Saturn's full run: 143 passed, 0 failed, including 106
regression tests across six registry and `calibrate_bar` consumers.

**`c1_propagate_t{1,2,8,32}`** — label `[n, s−t*]`,
`y_p = Σ_{h=1..t*} (Π a_i) b_{p−h}`. Registered in `M3_TASKS` and `E_T_STAR`,
routed in `e_hop_reading`. **No new channel** — reuses `CH_DRIVE`/`CH_FLIP`,
`d_model` unchanged, and `t*` rides in `f` (the chain head) so the dial
round-trips through a signature every consumer already passes.

**Nilpotent regime, chosen deliberately.** The sum terminates, so `t*` is an exact
hop count rather than a tolerance. That puts **both** the truncation law
`sqrt((t*−k)/t*)` **and** the flipper dependence `2√t*/(s−t*)` in closed form with
no fitted constant, and `k = t*+1` is **bitwise exact** — termination is shown,
not asserted.

### Saturn departed from the controller's literal spec, and was right to

The dispatch said the label is `[n, s]`. Saturn shipped `[n, s−t*]`:

> *"A strictly causal operator raised to `t*` vanishes on the first `t*`
> coordinates, always — causality and uniform depth cannot both hold on all `s`
> positions. `[n, s]` would ship `t*` entries of `sd 0`, the fourteenth strike in
> a new shape."*

`Ruling: ACCEPTED over the plan text. The controller's [n, s] would have shipped
constant-label positions, which is precisely the vacuity pattern this corpus
exists to avoid. The spec's binding requirement was a vector label whose every
position is informative, not a particular width. Cost if wrong: none identified —
the truncated width is the one consistent with causality.`

### VENUS'S PREDICTION SURVIVES THE NEW CORPUS, AND TIGHTENS

Her §6 demanded recomputation against whatever label shipped. Against C1's
weighting `w = 1/(s−t*)`, the predicted `settled` margins are

`−0.024827` / `−0.007715` / `+0.004809` / `+0.010545`

**All eight remain below the 13-seed resolution `0.027260`, and two are *smaller*
than what she filed.** These are `DERIVED` recomputations, not measurements —
nothing has trained on C1 yet (see below).

C1 removes her §2b and §2c but leaves **§2a untouched**. So the corpus does
**not** rescue the contract's optimism. What it buys is different and better:

> *"the label is uniform by construction, so a null is the arm"*

`Ruling: this is the round's central result so far and it is worth stating
plainly. The corpus was built to test the author's ambition and the pre-registered
counter-prediction still stands against it. But the old nulls were UNINTERPRETABLE
— confounded by a single-row write, a resolvent oracle, and a task softmax is
provably optimal on. A null on C1 is interpretable: it is the arm. Venus's
falsifier 5 collapses from a confound into a one-line test. The round has
converted an unanswerable question into an answerable one, which is what the
first nine iterations were for.`

### Saturn flagged the weak clause in his own bundle

The equal-variance clause **does not reject** the prefix scan at `t*=1` — reads
`0.0638`, inside his own `0.10` tolerance, because 63 of 64 positions are
uniformly *uninformative* rather than uniformly deep. The zero-hop clause rejects
at all four rungs. **Two clauses failing differently; either alone would have
admitted the worst rung.** That is the correct defence and he said so unprompted.

He also found **the shipped flipper band is too loose for this family**: exact
dependence `0.031746` at `t*=1` and `0.045620` at `t*=2`, both under
`bar_verdict`'s default `0.05` tolerance, so it would accept a flipper-blind
label at those rungs. Asserted as a test rather than left to be rediscovered;
anti-vacuity duty is carried by the movement test, which pins the exact bitwise
support.

### C1 is registered and calibrates but CANNOT TRAIN YET

`run_arm` raises `RuntimeError: The size of tensor a (64) must match the size of
tensor b (56)` — **loudly, not silently**. `56 = s − t*` at `s=64, t*=8`. C1 needs
Neptune's per-row readout to train, and Saturn correctly touched neither
`m3_quintuple.py` nor any weights.

**The integration of Saturn's corpus with Neptune's arm is the round's moment of
truth**, and both halves now exist or are in flight.

One shared function changed: `calibrate_bar`'s `payload_only` broadcast, verified
**bitwise-identical** for scalar labels (`1.366458892213088`,
`1.050147990558408`).

## Iteration 2 — Neptune COMPLETE. The arm exists and integrates with the corpus.

Merged. Controller integration check **48/48 green** across `tests/neptune/`,
Saturn's `test_c1_propagate_registration.py`, and
`tests/chase/test_pivot_exclusion_lift.py`. Neptune's own run: 116 passed.

### THE NUMBER THAT JUSTIFIES THE LANE

> **the shipped arm writes 1.6–3.1 % of the label's support; the per-row arm
> writes 88.1–100 %.**

Measured on untrained keys, so the fractions may drift. But that is the whole
argument in one line: the shipped arm was barely touching the thing it was being
scored on.

### Neptune corrected his own iteration-1 gate by 5×

His gate priced the per-row lane at **`≈29.6 h`** for ten units, from
`64 × 1.092959 s` — `_alpha` called once per row. The real arm **shares the Gram
and `A_P @ V`** and runs one Python loop over `[n·s, k]`. Measured: **`5.9 h`**.
Superseded in place.

His own diagnosis of the miss is the useful part: *"The gate's own limits
paragraph predicted this failure and was still not enough — the headline stated
an upper bound as a price."*

`Ruling: a headline number that is an upper bound must SAY SO IN THE HEADLINE, not
in a limits paragraph below it. This goes to Saturn for MISTAKES.md as a
provenance entry: the caveat was present, correctly worded, and still did not stop
the controller acting on the number — I ruled a 29.6 h pilot-only strategy on it.
Cost if wrong: nothing; the pilot was worth running regardless.`

### Integration: route 1, and his reason beat mine

The arm emits `[n, s]`; the slice lives in `_unit`'s existing `_A` adapter. I had
leaned route 1 on taste — *the arm should not need a corpus's difficulty dial*.
His argument is load-bearing instead: **`ROW_GATE_TERM` prices all `s` rows, so
route 1 keeps the FLOP accounting exact**, where route 2 would have over-priced by
`s/(s−t*)` — **2× at `t*=32`**.

The support is **checked, not inferred**: the offset from the label's width is
cross-checked against `e_t_star` and **raises on disagreement**. Both branches
tested — and **his first attempt at that test was vacuous** (the fake task carried
no dial) and **he caught it himself**.

### Two calibration facts that bind every clock in this round

1. **The box is `1.65×` slower than iteration 1 on identical code, geometry and
   seed** — the same `settled` unit read `2.077496 s/step` then and `3.437200`
   now. **No cross-iteration clock comparison this round is safe.** Only
   same-session ratios may be quoted.
2. **Pilot ratios do not transfer.** `settledrow/settled` is `3.48` at `s=16` but
   `7.4–11.2` at `s=64`, against a flat FLOP ratio of `1.700`. Neptune: *"It
   nearly caught me — I had drafted the `s=64` projection from the pilot ratio
   before measuring it."* The warning now lives inside the term.
3. **The pilot is not comparable to the shipped ladder in either direction.**
   `d = 8` is **forced** at `s = 16` — `make_batch` raises unless
   `1 <= d < s - 1`, so the shipped `d = 24` cannot be drawn. No pilot number may
   be set beside one from `results/e_ladder_reading.txt`. `k_piv = 8` was
   preserved deliberately: `batched_pivots` takes `min(k_piv, s-3)`, so `s = 16`
   is the smallest geometry that leaves the arm unchanged.

### Neptune refused to score Irene where his work cannot

- Her **falsifier 5 fires**.
- Her **PREDICTION 3 fails** — the RED gate reads `1.002360` at `t*=1`, not below
  `1.0`.
- Her **§2b is answered by Saturn's corpus, not his arm**.
- **PREDICTION 1's eight margins are NOT scored and cannot be from this work** —
  her §6 requires them recomputed against C1's *actual measured* variance profile
  first. His words: *"Nobody should record her as refuted on it."*

`Ruling: upheld, and the earlier ledger entry is amended. Saturn's recomputed
margins (-0.024827 / -0.007715 / +0.004809 / +0.010545) are DERIVED from C1's
construction, not measured from a trained run. Venus stands UNSCORED on
PREDICTION 1 until the deciding measurement exists. Recording her as refuted on a
derivation would be exactly the provenance failure this round has been correcting
in others.`

### Carried

- `PLUS_CELLS` break `eprocess._parse_key` — pre-existing, unfixed, and now
  relevant because the row cells join them.
- Neptune edited one existing test he broke: `test_pivot_exclusion_lift.py`'s
  `set(ALL_CELLS) == CELLS | PLUS_CELLS`, **which no third cell family could ever
  satisfy**. Intent preserved and strengthened to disjointness. Licit under the
  same exception granted to Saturn.
- The RED gate passes thinly — `0.999138` at `n_eval=64`.

### THE DECIDING MEASUREMENT IS NOW RUNNABLE, AND AFFORDABLE

From Neptune's measured table at `s=64`:

| cell | `n_train` | s/step | one unit |
|---|---|---|---|
| `settledrow` | 8192 | `25.543200` | `3831.5 s` = `1.06 h` |
| `settledrow` | **2048** | `4.560600` | **`684.1 s` = `0.19 h`** |
| `twinrow` | **2048** | `0.666200` | **`99.9 s`** |

`Ruling: the deciding measurement runs at s=64, d=24, n_train=2048, 5 seeds,
settledrow + twinrow — approximately 2.2 h, not the 10.6 h the n=8192 geometry
would cost. n_train=2048 is already the shipped ladder's own regime, so the
reading stays comparable to results/e_ladder_reading.txt in a way the s=16 pilot
explicitly is not. Assigned to Mercury, who holds execute. Cost if wrong: a
reading at 2048 rather than 8192 has wider intervals, and the pre-registered
13-seed resolution 0.027260 may not be reachable at 5 seeds - which must be
printed as a ceiling BEFORE the run, per LOOP_PROMPT 1.3.`

## Iteration 2 — Mercury COMPLETE. The statistics layer no longer dies mid-run.

Merged. Controller integration check **124/124 green** across `tests/mercury/`,
`tests/deimos/`, `tests/neptune/`, and the three `tests/chase/` suites covering
the modules touched.

### Unit 1 — the overflow, reproduced before it was fixed

`OverflowError: math range error` at `eprocess.py:313`, **crossed at draw 67,
crashed at draw 10135 of 10240**; an adversarial `+B` stream crashes at `1757`.

`log_value` / `log_peak` are now the primitives; `value` / `peak` saturate to
`inf` via `_exp_or_inf`; `crossed` compares in log space. This follows the
existing `log10_max_attainable` precedent — **every representable value is
bit-identical and no decision moved.** `max_peak`'s `min(max_log, 700.0)` clamp
fixed the same way. **No pre-registered constant moved**, and `update` still
raises on `|d| > B` rather than clamping.

**The load-bearing half:** `calibrate` now runs **the shipped class** over 8
replays per calibration and refuses at `BIND_TOL = 1e-9`. Worst real gap
`3.553e-15`, and **the check is shown to read FALSE under a drifted `update`** —
so the battery finally exercises the object production uses, which was Deimos's
actual finding.

### Unit 2 — the table cut, with the guard verified first

Neptune's guard was checked **before** the cut and **against the real
`ceq/hf_artifact`, not a fixture**. Manifest survived, all five checkpoints
named, neither denial string returned. v1 and the HF artifact cut;
`results/capability_table_v0.*` byte-identical.

### Mercury caught Neptune's repair shipping NEW wrong numbers

Neptune's repaired `LIMITS` clause shipped **three stale numbers** — *"60 of the
100 rows … 15 each"* against a journal holding **61 of 86** with `e3_t1` at
**16**. Two were true at `25b9cb8`; **`100` never was.** Now derived by
`_task_census` at cut time rather than stored.

**And a second clobber was still armed:** the plain CLI wrote the frozen v0.
`BOARD.md:182-191` had documented a hand-edit workaround instead of fixing it.
Mercury added `--version`, defaulting to `v1`.

`Ruling: a repair that hardcodes fresh numbers is the same defect it repaired.
Mercury's fix - derive the census at cut time, store nothing - is the correct
shape and generalises. This is the second time this round a planet has caught the
previous planet's fix rather than the original bug, and both catches were right.`

### TWO INDEPENDENT PRICINGS DISAGREED. Resolved in Neptune's favour.

- **Neptune: `5.9 h`** for ten units — a `RUN` measurement of the shipped
  implementation (`settledrow` `25.5432 s/step` → `1.06 h`/unit; `twinrow`
  `2.7116` → `0.113 h`/unit; `5 × (1.06 + 0.113) = 5.87 h`).
- **Mercury: `15.5 h`, `13.5×`** — a corrected *analytic* projection. He confirmed
  Neptune's `settled` base rate independently (`311.62` derived against `315.65`
  journal median, ratio `1.013`) and correctly showed the original gate charged
  **all ten units at the settled rate** when `m3_quintuple.py:293-295` makes
  `twin` a single `logsumexp` with `need_gram=False` — no settle loop for the ×64.

`Ruling: Neptune's 5.9 h stands as the operative number. Both planets are right
about different objects: Mercury corrected the arithmetic of the ORIGINAL 29.6 h
gate, which priced a hypothetical implementation calling _alpha once per row;
Neptune replaced that implementation with one sharing the Gram and A_P @ V, then
measured it. A measurement of the shipped thing outranks a corrected estimate of a
thing nobody built. Mercury's correction is not wasted - it shows the original was
wrong two independent ways, implementation AND per-arm rate. Cost if wrong: the
deciding run takes longer than budgeted, which the bucket budget absorbs.`

`Ruling: MERCURY'S CONCERN 6 IS A MISTAKE TYPE WITH THREE INSTANCES and goes to
Saturn for MISTAKES.md - "pricing every arm at the dearest arm's rate". Seen at
STATE.md:21 (over-priced the ladder 3.0x), at r9_systems_gate.md:196 (the 29.6 h
headline), and in the original ladder estimate. Three independent occurrences make
it a pattern, not a slip.`

### Carried, and one adjudication owed

- **`impact` / `impact_hetero` remain in `M3_TASKS`** and therefore in the
  freshly cut capability card's `registry`, despite being retired on three
  independent kills and despite `MISTAKES.md:215-228`. Mercury correctly declined
  — deregistering is an adjudication, not a runner's call.

`Ruling: impact and impact_hetero are NOT deregistered, but the published card
must stop advertising them as admitted tasks. Deregistration would break Saturn's
CH_HET repair, Mercury's A5/A6 fixes and their tests, all of which are correct
work on a task that simply has not earned admission. The honest state is: present
in the registry, marked NOT ADMITTED with the three kills cited. Assigned to
whoever next holds the card. Cost if wrong: the card carries a row with a caveat
instead of no row.`

- Mercury converted one of Deimos's bug-asserting tests now that the bug is fixed
  — docstring verbatim, assertions inverted, old name carried for ledger
  traceability. He flagged it because it edits another agent's file. **Correct
  disclosure and correct handling.**
- The twin/settled ratio **does not carry across tasks**: twin is `1.61×` *dearer*
  at `negation_scope`/ntr8192 and `2.1×` cheaper at `e3_t1`/ntr2048.

## Iteration 2 — Mars COMPLETE. The STE cell exists; the question is not yet answered.

Merged. Controller integration check **70/70 green** across `tests/mars/`,
`tests/chase/test_pivot_exclusion_lift.py`, `tests/neptune/`, and Saturn's C1
bundle. **Four cell families now coexist**: `CELLS`, `PLUS_CELLS`, `ROW_CELLS`,
`STE_CELLS`.

`argmaxste`, estimator `hard + (soft - soft.detach())`. Forward is bitwise the
one-hot (`x - x` is exactly `+0.0`); backward is the softmax Jacobian.

**The parentheses are load-bearing** — `hard + soft - soft.detach()` rounds first
and is **not** bitwise. Mars made that RED 2 and asserted it by value on a
4096×8 float64 draw. That is the kind of detail that silently invalidates a bind.

Binds, each with its own RED per `arm_s.py:341-346`: `torch.equal` against
`argmax` **3/3** at (8,0)/(16,1)/(32,2); RED 1 `twin` vs `argmax` **3/3 not
equal**; step-0 loss identical at `1.108632`; **gate gradient equals twin's to
rel `1e-12` while `argmax` stays exactly `0.0`**; `n_params == 4769` for all five
cells. So the two cells differ in *exactly one thing*: whether selection is
trained.

Price, analytic only, **zero clock quoted**: `argmaxste` **7,367,294,976** FLOPs,
equal to `twin`'s, above `argmax`'s `7,365,197,824` by exactly
`2,097,152 = n·2·k·d_model` — the contract term `argmax`'s gather does not pay.

### Mars's own first statistic failed vacuity rule 6, and he reported it

His slot-drift statistic was confounded: the pivot **set** turns over **256/256 in
both cells** over 30 steps, because `select_pivots` is top-k over `kk.norm()`.
Measured `argmax` 200/256 against `argmaxste` **183/256** — **the estimator moved
fewer**. He discarded it as a claim rather than shipping it.

The clean replacement is parameter divergence from identical init and batch:
**`3.871934` on `‖θ‖ = 11.981715`, 32.3 %.**

This also **sharpens his own iteration-1 attack**: the choice *among the selected
candidates* gets no gradient; the candidate set itself does move, through `wk`.
That is a narrower and more defensible claim than the one he originally filed.

`Ruling: a planet narrowing its own earlier claim on new evidence is exactly the
behaviour the adversary seat exists to produce, and the ledger records the
narrowed version as operative. The iteration-1 wording stands corrected, not
withdrawn - argmax's alpha still carries no grad_fn, and the headline -0.118456
still confounds two things. What changed is the precise statement of which two.`

### The question remains OPEN, and Mars said so plainly

**No execute seat, no NRMSE taken.** His report carries pre-registered rows **A–E**,
exhaustive before any data:

- **Row D is live** — a straight-through gradient is biased and can make the cell
  *worse*, which would license nothing.
- **Row E is live** — if `argmaxste` reads above `1.0` like `argmax`'s
  `1.010779`, the contrast is **uncreditable by the repo's own rule regardless of
  direction**.
- The 30-step loss pair (`1.035080` vs `1.033804`) is **far inside bootstrap zero
  and must not be quoted as a direction.**

### Pre-existing, found not repaired

`_bind_batched_against_arm_s` returns **`False` on its DIAGNOSTIC shapes** at
clean `5201d78` — `n=2, s=128, d=16, k=32, seed=2`, max abs diff `1.387779e-17`.
Confirmed pre-existing by stashing. **Does not block the run**: `main()` gates on
the run's own shapes, which bind **BITWISE 2/2** at `s=64, d=24, k=8`.

## Iteration 3 — dispatched

| Planet | Unit | Status |
|---|---|---|
| Mercury | **Run the STE reading** on `negation_scope`, against Mars's rows A–E, ceiling printed first | dispatched |
| Venus | Nine ghost symbols in `test_harmonic_attribution.py`; file her prediction for the C1 run | running |

`Ruling: THE C1 DECIDING MEASUREMENT IS HELD until Venus files. It is priced,
cleared and ~1.1 h of compute, and I am deliberately not starting it. Her seat
exists so insight arrives ex ante where it can be scored; running the deciding
measurement before her filing lands would destroy the only thing that makes the
filing worth anything. Cost if wrong: roughly one agent-turn of latency on the
round's most important number.`

## Iteration 3 — Venus COMPLETE. Holding the run for her paid for itself immediately.

Merged. `tests/cameron/test_harmonic_attribution.py` **11 failed → 1 skipped**;
`tests/deimos/` 8/8. The `house-events.jsonl` union driver worked — auto-merged,
no conflict.

### The producerless clause is STRUCK, and the proof is the good part

`git log -S` returns **zero commits across all refs** containing a definition of
any of the nine producers. The pilot numbers had **no possible source in any
state this repository has ever been in** — case 3, not deletion, not
sibling-transcription.

**And she proved her search works before trusting its zero** — vacuity rule 6,
applied to her own instrument: planted positives `def harmonic_measure` →
`22036de` at `kirchhoff.py:187` (a different object) and `def rag_document_ids` →
`74e5590`. A search returning zero that was never shown to find a witness is
mistake type 12, and she did not commit it.

Two further findings turned *"write the producers"* into the wrong answer:
- **The floor's provenance is a units change.** `BOARD.md:271` says the bar is
  `PASS_BAR = 0.5` *inherited unchanged* — but that is an **NRMSE** bar at
  `rips_gate.py:60`, re-used here as a **Spearman rho** floor.
- **`K-R8d` is already taken** and answered as the decoder-leak kill
  (`BOARD.md:297`).

Struck per `CHECKLIST.md:478,486,665`. Three constants into the `STRUCK` registry
(9→12, must-fire verified **both** directions), **302 lines kept verbatim**, and a
module-level `pytest.skip` on a live `hasattr` check — deliberately **not**
`xfail(strict=True)`, which would file these eleven alongside the ~50 that *are*
the record.

Her hard-rule-2 argument, stated narrowly as required: no definition in any ref,
zero branches executing (0.44 s, all `AttributeError`), no xfail machinery in
`tests/cameron/conftest.py`, and the finding already independently held by
`tests/deimos/...:349`. **"It generalises to nothing whose symbol has ever been
defined."**

`Ruling: the strike is ACCEPTED and the hard-rule-2 exception is granted on her
stated grounds. The eleven are not a finding reproducing - they are a claim whose
evidence never existed. Her narrow framing is what makes the exception safe: it
cannot be stretched to cover any test whose subject exists.`

### SHE FOUND A FLAW IN THE CONTROLLER'S OWN RUN SPEC — three of them

**1. I omitted `softmax` from the deciding run.** `settledrow − twinrow` is the
settling contrast and **cannot address the `arXiv:2410.01537` Bayes-optimality
claim that justifies the entire lane**. `softmax` under `vector_readout` is the
cheapest cell on the board — `0.006437 s/step` against `settledrow`'s
`4.560600` — and without it at matched seeds **`twinrow − softmax` is
unavailable forever after**. Amended: the run is `softmax + twinrow + settledrow`.

**2. At N=5 the verdict is a sign test.** Measured on the shipped `contrast()`
over 1,000 samples: unanimity excludes zero **385/385**; a 4–1 split 20–44 %; a
3–2 split 0–3.7 %. So *"the CI excludes zero"* at five seeds is very nearly
*"all five seeds agreed"*, and **the finest achievable two-sided p is `0.0625`,
not `0.05`.**

**This is a property of every 5-seed reading in the repository, including the
standing `+0.108437` headline.** Amended: print `n+` beside every CI.

**3. `t*=1` is underpowered by an order of magnitude.** Realised `sd_paired` is
**0.019–0.109**, not the pilot's `0.050`; `t*=1` needs **62 seeds**, not 5.

`Ruling: run 5 seeds on all four rungs with softmax added; print the per-rung
ceiling BEFORE the first number. Do NOT run 62 seeds on t*=1 - ~13.5 h for one
rung against ~4.4 h for the whole table. Do NOT drop it either: dropping the rung
that cannot resolve and reporting only the rungs that can is rung-picking, the
exact defect M4 exists to prevent. Run it, label it UNDERPOWERED with its 62-seed
requirement in the same breath, and show the whole ladder. Cost if wrong: one rung
carries a caveat instead of a verdict, which is what it has earned.`

### Her prediction, filed before the data

> `settledrow − twinrow` reads inside `±0.027260` at every rung with every CI
> covering zero — **the write share moves 56× and the settling contrast does not
> move.**

Point estimates are the scalar ladder's own four deltas. She flags honestly that
this **transfers** the `e3` deltas onto `c1_propagate` under a different readout —
a transfer, not a derivation — and falsifier 4 covers the spread half. Four binary
falsifiers plus fall-through row **Ω2**.

This is the bet worth watching: Neptune measured write share going from
**1.6–3.1 %** to **88.1–100 %**. If a **56× change in write share moves nothing**,
that is a result about the arm rather than about the corpus — which is exactly
what C1 was built to make sayable.

### Carried, for another owner

`scale/chase_struck_coverage.py` reports **"SCANNING 0 PATHS"** and *"uncovered
.md: 0, uncovered .py: 0"* against its own docstring's claim to scan every
uncovered file. **Empty coverage** — which is why Venus's registry addition had no
collateral. A hole in a different instrument, left for its owner.

## Iteration 3 — Saturn COMPLETE. `MISTAKES.md` 31 → 35, and he corrected four of the controller's numbers first.

Merged. Controller check **36/36 green**.

New entries: **P-8** upper bound stated as a price; **M-8** pricing every arm at
one arm's rate; **M-9** finest achievable p cannot reach its α; **V-13** a search
whose walk includes nested checkouts. **M-3** absorbed the cost and power faces
of pilot-transfer. V-13 was **appended rather than inserted** so nothing
renumbers — the five files citing `V-1/V-8/D-1/D-4/M-2` all still resolve.

### He refused to ship four of the dispatch's own numbers

1. `m3_quintuple.py:293-295` for `need_gram` is **drifted** — the real sites are
   `:404` / `:444`.
2. *"Nine worktrees"* — `git worktree list` returns **ten** (the tenth is the
   primary checkout), and the phantom count has already moved `10 → 9` since
   Deimos measured it.
3. **He could not source M-8's third instance as a distinct document.** Mercury
   *counts* three; Saturn's entry evidences two and **attributes the count to
   Mercury rather than padding a bullet to reach three.**
4. Neptune's pilot quote as the controller gave it was a **paraphrase**; the
   entry uses Neptune's actual words.

`Ruling: refusing to pad M-8 to three sourced instances is the correct call and
the ledger records it as such. A mistakes file that inflates its own evidence
count would be its own first entry. The controller supplied the paraphrase and the
drifted line numbers; that is the same defect FINDINGS section B was demoted for,
recurring in dispatch prose rather than in a findings table.`

### He added a citation binder for the file that documents citation failures

`tests/cameron/test_mistakes_citations_resolve.py`, 7 tests, scope-adjacent and
justified: **both failure classes `MISTAKES.md` documents have already happened to
`MISTAKES.md` itself** — four citations wrong on first write in iteration 1, two
drifted by the same commit's own edits, and `mercury-report.md:49` broken this
iteration. **The checker caught that one pre-commit, which is its RED.**

Stated limit: **the binder checks locations, not content.** A cited line that
still exists but now says something else passes. Closing that needs each entry to
carry a re-matchable quoted fragment; not built.

### THE HEADLINE INTERVALS DISAGREE, AND IT IS A PATTERN ACROSS ALL THREE CONTRASTS

Saturn found one instance. Controller verification (`READ`, this session) found it
is systematic — **two entire families of intervals, every upper bound agreeing and
every lower bound differing**:

| contrast | family A — ships in `ceq/hf_artifact/README.md` + `capability_table_v0/v1` | family B — `CHECKLIST.md` + `DONE.md` + root `README.md` |
|---|---|---|
| `settled − twin` | `[-0.048587, +0.031557]` | `[-0.042903, +0.031557]` |
| `settled − softmax` | `[+0.066232, +0.147110]` | `[+0.068181, +0.147110]` |
| `argmax − softmax` | `[-0.134115, -0.102204]` | `[-0.134115, -0.102786]` |

**`DONE.md` contains both families** — `:1844` and `:2198` carry family B,
`:1884` carries family A. So both were journalled, then propagated into different
documents.

**The likely cause, stated as a hypothesis and not yet confirmed:**
`CHECKLIST.md:1168` describes its interval as *"exact enumeration over all
`5**5 = 3125` paired resamples"*, while `m3_synthetic_settled.contrast` runs an
`n_boot=10000` percentile bootstrap. **Two legitimate procedures, both correct,
producing systematically different lower tails — and neither document names which
one produced its number.**

If that is right, this is not a bug in either number. It is a **provenance
failure on the project's most-quoted result**, and the version that **ships to
HuggingFace** is not the version the repository's own README quotes.

`Ruling: routed to Saturn, who found it and owns the instrument class. He
deliberately left both in place - "picking one without finding the producer makes
the disagreement invisible" - and that was right. The fix is to find the producing
run for each family, name it beside every published interval, and make the
artifact and the README agree on which procedure ships. Cost if wrong: the two
numbers stay side by side with their provenance attached, which is still better
than one silently chosen.`

### Stated limit on everything Saturn shipped

**He re-measured only the two cheap things** (worktree count, live phantom count).
The `3.0×`, `5×`, `1.61×`, `2.1×`, `15.5 h` and `5.9 h` figures in M-8 and P-8 are
all **READ, not RUN** — if Mercury's or Neptune's timings are wrong, those entries
inherit the error and would not detect it.

## Iteration 3 — THE STE READING. The project's headline explanation does not survive.

Merged, **72/72 green**. `tests/mercury/` 50, `tests/mars/` 22.

| cell | mean NRMSE |
|---|---|
| `argmax` | `1.010779` — **worse than predict-the-mean** |
| `softmax` | `0.892323` |
| **`argmaxste`** | **`0.785019`** |
| `twin` | `0.780927` |

| arm | vs | delta | CI | `n+` |
|---|---|---|---|---|
| `argmaxste` | `argmax` | **`+0.225760`** | `[+0.212433, +0.245886]` | **5/5** |
| `argmaxste` | `softmax` | **`+0.107304`** | `[+0.082879, +0.140870]` | **5/5** |
| `argmaxste` | `twin` | `−0.004092` | `[−0.029187, +0.023107]` | 2/5 |
| `argmax` | `softmax` | `−0.118456` | `[−0.134115, −0.102204]` | 0/5 |

Mars's pre-registered rows scored verbatim: **A False, B False, C TRUE, D False,
E False.**

### THE ANSWER IS THE SELECTION, NOT THE MIXTURE

A one-hot lookup whose **selection is trained** moves from `1.010779` — worse than
predicting the mean — to `0.785019`, **past softmax**, to **statistically
indistinguishable from the full mixture**. Training selection moves the cell by
**190.6 % of the disputed gap**.

The forward function never changed: `argmaxste` is **bitwise identical** to
`argmax` in the forward. The only difference is that gradient reaches the gate.

`Ruling: the README's "The gain is the mixture, not the equilibrium" DOES NOT
SURVIVE and must be withdrawn on its mixture half. Mars's iteration-1 attack
showed the -0.118456 confounded mixture-vs-lookup with trained-vs-untrained
selection; this reading resolves the confound and the answer is selection. The
EQUILIBRIUM half is untouched - no settled cell was in this reading - so the
sentence is not wholly false, it is half-unsupported and the wrong half was the
one being sold. Cost if wrong: a published claim is withdrawn that a later
settled-inclusive reading might have restored.`

**Row E was live and is closed by measurement**: `argmaxste` clears
predict-the-mean by `0.214981`, so the contrast is creditable rather than void.
And the last row **reproduces the disputed headline to six decimals on the same
seeds** — a control on the pipeline itself, not just on the cells.

### Mercury's three stated limits, all correct

1. **The `argmaxste − twin` null is a BOUND, not an identity** — bounded at
   `0.026147` by this run's own CI half-width. *"No difference larger than
   `0.026147`"*, never *"identical"*.
2. **`verdict_of` prints `SETTLED WINS` / `TWIN WINS` for any contrast**, because
   it was written for that one pair. On an `argmaxste`/`softmax` row those strings
   **name no cell**. He relabelled in his report and quoted none of them. With
   four cell families now live this function is actively misleading — **queued**.
3. **No same-session `twin`**, so `argmaxste`'s median `221 s` cannot be ratioed
   against the journalled `twin` median `508.74 s`. Whether it is genuinely
   cheaper or the box was fast is **unresolved and unclaimed**.

## Iteration 4 — Saturn CONFIRMED the interval hypothesis exactly

Merged, **19/19 green**. New binder `tests/cameron/test_published_intervals_have_producers.py`, 12 tests.

**Family A is `contrast(n_boot=10000, seed=0)`. Family B is exact enumeration
over all `3125` paired resamples** (126 distinct means, measured). Reproduced
**twice** — on the 6-dp printed values (5 of 6 bounds exact, the sixth off by
`1e-6`, which is input rounding) and on full-precision journal floats (**all six
exact**).

**No published interval is orphaned. No number is wrong.**

### But the real finding is worse than provenance

Saturn: *"it's P-8 recurring, on the most-quoted result, while P-8 was being
written."*

It was **already known and written in four places, including the shipping
artifact** — limits paragraph (e) at `ceq/hf_artifact/README.md:80` states the
whole thing correctly. **The table three lines above it still prints `95% CI`
with no estimator, and the JSON behind it already carries
`n_boot: 10000, boot_seed: 0`.** The provenance exists in the data and is
**dropped at render time**.

That is P-8 exactly — the caveat present, correctly worded, in a limits
paragraph, while the headline drops it — occurring on the same page as the entry
being written about it.

### Saturn corrected the controller's framing of his own finding

My dispatch said the artifact and the root README disagree silently. **Root
`README.md` was already correctly labelled** (`exact 95 % CI`, preamble naming
the enumeration) — nothing to do there. Chase's test already bound family A. The
only genuinely unbound half was **family B, which existed solely as transcribed
digits in three prose documents** — *"that's how an estimator came to look like a
discrepancy."*

`Ruling: my "the shipping artifact disagrees with the README" framing was wrong
and is withdrawn. The README was labelled; the artifact's TABLE was not, though
its own limits paragraph and its JSON both were. Two text-only changes are owed
to Mercury, who holds those files: render() should emit the two fields the JSON
already holds (column header -> "95% CI (percentile bootstrap, B=10000, seed=0)"),
no recomputation needed; and limits paragraph (e) is itself now stale, carrying
+0.146551 as an open defect where CHECKLIST.md:1240 and README.md:460 both record
that endpoint corrected to +0.147110.`

**Stated limit:** Saturn reproduced **the procedures, not the run**. If a per-seed
value in the journal is itself wrong, both families inherit it identically and
every new assertion still passes. He also read the committed blob rather than the
working file, which is under active write by Mercury's C1 run.

## C1 status — in flight

`t*=32` softmax weights are landing in the journal now; rung order `t32 → t8 →
t2 → t1`. No C1 number exists yet and none is claimed.
