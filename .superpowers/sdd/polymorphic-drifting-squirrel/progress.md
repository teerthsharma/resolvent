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
