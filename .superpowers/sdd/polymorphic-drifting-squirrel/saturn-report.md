# SATURN (WATSON) — iteration 1 report

Role: instrument-keeper. Calibration, must-fires with non-degeneracy checks,
the bind law. No claim below entered a verdict except through a test that was
RED first, and both RED runs are quoted.

**Status: DONE_WITH_CONCERNS.**

---

## 0. Worktree correction, first, because it changes what "this tree" means

The dispatch said branch `feat/r9-causal-consequence`. The worktree
`C:\Users\seal\Desktop\New folder (32)\.claude\worktrees\agent-a02edd6dcdc5b62d7`
was on `worktree-agent-a02edd6dcdc5b62d7` at `ac47049`, which is an **ancestor**
of `74e5590` (`git merge-base HEAD master` returned `ac47049` itself).
`scale/impact.py` did not exist in it. `feat/r9-causal-consequence` is checked
out in the primary worktree and cannot be checked out twice, so the branch was
fast-forwarded to `74e5590` — the exact commit FINDINGS audits — with
`git merge --ff-only`. Everything below is against that tree. Work is committed
on `worktree-agent-a02edd6dcdc5b62d7`, not on `feat/r9-causal-consequence`; it
needs a merge or cherry-pick to reach the round's branch.

---

## 1. Claim ledger

| # | Claim | Class | check: |
|---|---|---|---|
| 1 | `scale/negation_scope.py:1081` bound `impact_hetero` to `make_impact_batch`, the same object `impact` binds | READ | `grep -n "impact" scale/negation_scope.py` → `1077` and `1081` name the identical `fromlist=["make_impact_batch"]` import |
| 2 | The two keys returned a byte-identical tensor | RUN | RED run: `AssertionError: the two registered keys draw a BYTE-IDENTICAL tensor` |
| 3 | Both builders refuse `s < 1024` | READ | `scale/impact.py:596-597` and `:633-636`, `IMPACT_MIN_NODES = 1024` at `:73` |
| 4 | The two builders genuinely produce different corpora | RUN | at `s=1024, n=8, seed=0`: label differs 8/8, `max|x_homo − x_het| = 1.712538`, query node `673` vs `866` |
| 5 | Neither label is degenerate | RUN | label sd `2.478796` (impact) and `2.735918` (impact_hetero), both finite |
| 6 | Rebinding the builder ALONE would have shipped a mislabelled task | RUN | every consumer hardcoded `heterogeneous=False`; with the plant bit erased the registered oracle reads NRMSE **`1.399930`** against its own label — worse than predict-the-mean |
| 7 | After the repair the registered oracle reproduces its builder's label | RUN | `nrmse(oracle, y)` = `2.289e-08` (impact), `3.268e-08` (impact_hetero), both `< 1e-6` |
| 8 | The repair did not disturb the `impact` corpus | RUN | sha256 of `x` and `y` at `n=4, s=1024, seed=0` identical pre- and post-repair: `5cb97ab298adc015` / `46ad483cab16c909`, `f=673`, `p=0` |
| 9 | `calibrate_bar` trains on raw `y`; `run_arm` trains on standardised `y` | READ | `scale/negation_scope.py:1195-1196` `loss = ((net(feats).squeeze(-1) - y) ** 2).mean()` vs `scale/m3_capability.py:196` `y_train_std = (y_train - mu) / sigma` and `:200` `mse_loss(model(x_train), y_train_std)`; `raw_pred` un-standardises at `:189` |
| 10 | The e2 bar reads `2.446645` at `steps=150` and calibrates at 600 | RUN | reproduced at `n=2048, s=64, d=24, lr=0.02`; `STATE.md:73-76` records `2.446646` |
| 11 | The control's reading was a function of the label's UNITS | RUN | RED run: scaling the label by `0.1` moved `trained_two_feature` from `2.446645` to `23.272818`, gap `20.826173` |
| 12 | The repair makes the reading scale-free | RUN | after: gap `0.000000` at both `c=10.0` and `c=0.1` |
| 13 | The repair calibrates the e2 bar at the shipped budget | RUN | `2.446645` BROKEN → `0.699440` CALIBRATED, same 150 steps, same batch |
| 14 | Clauses 1-4 are bit-identical across the repair | RUN | `predict_the_mean`, `payload_only`, `oracle`, `flipper_dependence` compared with `==`, all equal |
| 15 | The shipped `negation_scope` task still calibrates | RUN | `trained_two_feature = 0.046744`, BAR CALIBRATED |
| 16 | `tests/cameron/test_m3_etasks.py` was 34/34 before my change | RUN | baseline run on the stashed tree: `34 passed in 6.09s` |
| 17 | I caused exactly one failure and it was the defect-lock test | RUN | post-change: `1 failed, 33 passed`; the failure is `test_the_consequence_bar_is_broken_at_the_shipped_step_budget` asserting `not ok150` |
| 18 | The 11 `test_harmonic_attribution.py` failures pre-date me | RUN + READ | `AttributeError: module 'scale.negation_scope' has no attribute 'absorbing_boundary_kernel'`; `git show 74e5590:scale/negation_scope.py \| grep -c absorbing_boundary_kernel` → `0`; my diff contains 0 occurrences |
| 19 | FINDINGS B4 is **partly wrong** | READ | see §5 |
| 20 | "the fifteenth vacuous control" is an already-claimed label | READ | `DONE.md:1228` and `PIVOT_EXCLUSION_FALSIFIER.md:50-52` each claim it for a different item; `STATE.md:53` states fourteen struck |

Adversarial pass, per claim class. Claim 4 would pass with the repair deleted
only if the builders coincided — they do not, and claim 2's RED run proves the
registry did not use them. Claims 4 and 7 do not share an assumption: a repair
that bound the hetero builder and left the oracle homogeneous makes 4 pass and 7
fail, which is exactly what claim 6 measures. Claims 11/12 (a property over a
family of labels) and 13 (one absolute number on record) fail differently: a
standardisation-arithmetic bug breaks the first, a wrong scoring scale breaks
the second.

---

## 2. Deliverable 1 — `MISTAKES.md` at the repo root

Written. 22 entries in four classes, each carrying the type name, the real
instance with its citation, and the design rule that prevents a repeat.

- **V, vacuous controls, 12 entries.** V-1 two registry keys / one corpus, V-2
  hand-built minimal example, V-3 algebraic identity, V-4 fired on the wrong
  cause, V-5 slice-to-nothing, V-6 branch never ran, V-7 a search incapable of
  finding anything read as absence, V-8 constant PASS-half label, V-9 a repair
  that changes nothing, V-10 a gate satisfied by construction, V-11 a
  precondition satisfied at every draw, V-12 a single absorbing target.
- **P, provenance, 7 entries.** No live producer; number whose only home is a
  commit message; stale claim never retracted; claimed scaffolding that does not
  exist; doc rot; line-reference drift; vocabulary with no referent.
- **M, measurement, 7 entries.** Train/eval preprocessing mismatch; threshold
  refitted to the data it judges; pilot spread taken as realised; single-seed
  bootstrap read as seed variability; a process that cannot cross its own
  threshold; a partial run read as a verdict; a pre-registration with a hole.
- **D, design-level, 5 entries.** D-1 racing a baseline at its proven optimum is
  given the most prominent treatment and is named as subsuming most of the null
  results; D-2 an oracle that is the arm's own resolvent; D-3 a difficulty dial
  that does not vary; D-4 registration without admission; D-5 declaring `0.0`
  without a movement test.

It closes with the seven pre-ship checks, each pointing back at the entry that
earned it, plus D-1 as the eighth.

**Two substantive departures from the dispatch's list**, both because the
evidence did not support what was described.

1. **"Identical-twin registration" is not on record as a strike.** An exhaustive
   grep for `identical.twin`, `twin registration`, `its own twin`, `twin of
   itself` across every `.md` and `.py` returns four hits, none of which is a
   struck control. The nearest real object is the opposite: `rag_multihop_t*`
   registers the e3 corpus a second time under document-graph naming and is
   **licit**, because it says so — `scale/negation_scope.py:907-921` reads *"the
   rag reading is a NAMING of the same corpus object, not a second corpus"* and
   *"IDENTICAL TENSORS IS TESTED, NOT STATED"*, asserted bitwise in
   `tests/cameron/test_u1_rag_registration.py`. V-1 is written around that
   contrast, which sharpens the rule: the defect is not two keys sharing a
   corpus, it is an entry **claiming** a different corpus and not being made to
   prove it.
2. **The ordinal "fifteenth" is not available.** `STATE.md:53` states fourteen
   struck. `DONE.md:1228` and `PIVOT_EXCLUSION_FALSIFIER.md:50-52` each already
   claim "the fifteenth" for different items. `MISTAKES.md` therefore calls the
   count a floor and claims no ordinal for A1; the code comment and test
   docstring were edited to match after I first wrote "the fifteenth" into both.

Citations were checked mechanically: 62 numbered `file:line` references and 16
bare file references verified for existence and range against the working tree.
Four of my own citations had drifted — two because my edits shifted lines in the
files I was citing, which is precisely entry P-6 — and were corrected before
commit.

---

## 3. Deliverable 2 — the live vacuous control (FINDINGS A1) is dead

**RED first.** `tests/cameron/test_impact_hetero_is_not_its_own_baseline.py`,
six tests, run before any source change: **3 failed, 2 passed in 20.54s.**

```
E   AssertionError: the two registered keys draw a BYTE-IDENTICAL tensor
E   assert not True
E    +  where True = torch.equal(tensor([[[-0.4238, 0.2562, ...]]]),
E                                tensor([[[-0.4238, 0.2562, ...]]]))

E   AssertionError: the two plants produce identical local features on 8 of 8 rows
E   assert 0 == 8
```

The two that passed at RED are the guards against a bad repair — the label was
non-degenerate and the oracle matched *because both keys were the homogeneous
task*. They are what makes a naive rebinding fail.

**Admission checked before sizing the draw**, as instructed. Both builders raise
below `IMPACT_MIN_NODES = 1024` (`scale/impact.py:73`, enforced at `:596` and
`:633`), so every draw is at `s=1024`, `n=8`, module-scoped because a graph
build costs about 2.7 s.

**The repair is not the one-line rebinding.** Reading the consumers first showed
that `impact_oracle`, `impact_oracle_vec`, `impact_features`,
`impact_planted_features` and `impact_truncation` all rebuilt the graph with
`build_impact_graph(s, graph_seed, heterogeneous=False)` hardcoded. Binding
`make_impact_hetero_batch` alone would have traded a vacuous control for a
**mislabelled** one: `calibrate_bar` clause 3 requires
`nrmse(oracle(x,f,p), y) < 1e-6`, and the homogeneous kernel against the
heterogeneous label reads **`1.399930`**. So:

- `CH_HET = 15` — the previously unowned channel — carries the plant bit, 0.0
  homogeneous / 1.0 heterogeneous, written by `make_impact_hetero_batch`.
- `_graph_from_x(x)` rebuilds from `(shape, CH_SEED, CH_HET)`. All five
  consumers route through it. The bit rides in the tensor rather than in the
  registry, so **no future entry can pair a builder with the wrong oracle** —
  the fix is one guard where every caller already passes, not five.
- `scale/negation_scope.py` now binds `make_impact_hetero_batch`.

**GREEN: 6 passed in 24.89s.** Discriminates on **8/8** drawn instances,
`max|x_homo − x_het| = 1.712538`, query node `673` vs `866`, label sd
`2.478796` vs `2.735918`, oracle `2.289e-08` / `3.268e-08`, features differ on
8/8 rows.

**Non-degeneracy and the load-bearing check, both shipped.** `sd > 0` and
finiteness asserted on both labels; and per the vacuity rule's clause 6 the
repair is deleted in-process — erase `CH_HET` on a live tensor and the oracle
goes `3.268e-08 → 1.399930`. If that number did not move, `CH_HET` would be
decoration.

**Blast radius, measured.** The `impact` corpus is byte-identical before and
after: sha256 `5cb97ab298adc015` (x) and `46ad483cab16c909` (y) at
`n=4, s=1024, seed=0`, same `f=673`, `p=0`.

---

## 4. Deliverable 3 — the `e2_consequence` bar (FINDINGS A7)

**Both call sites read and quoted**, as instructed:

```
scale/negation_scope.py:1195-1196   (calibrate_bar clause 5)
    loss = ((net(feats).squeeze(-1) - y) ** 2).mean()

scale/m3_capability.py:196, :200    (run_arm)
    y_train_std = (y_train - mu) / sigma
    loss = torch.nn.functional.mse_loss(model(x_train), y_train_std)
```

and `run_arm` un-standardises before scoring — `return out_std * sigma + mu`,
`scale/m3_capability.py:189` — so its *reported* number is on the raw scale while
its *optimisation* is on the standardised one. The control did neither, and both
were then compared against the same bar of 1.0.

**The mechanism, measured not assumed.** Adam's step is bounded by `lr` almost
independently of the gradient, so the distance from the control's
initialisation (`normal_(weight, 0.0, 0.5)`, output order 1) to the label's
scale is spent out of the step budget. `e2_consequence`'s label at the house
shape has mean `0.000717` and **sd `0.061984`** — the offset is negligible, the
*scale* is the whole defect.

**RED first.** `tests/cameron/test_bar_control_sees_the_arms_preprocessing.py`,
five tests: **4 failed, 1 passed in 5.30s.**

```
E   AssertionError: scaling the label by 0.1 moved the trained control from
E   2.446645 to 23.272818 -- the control is optimising on the label's raw units
E   while run_arm optimises on the standardised label
E   assert 20.826173149216554 < 0.001
```

The one that passed at RED is `test_the_shipped_negation_scope_task_still_calibrates`
— deliberately, because it is the guard that the repair does not move the
defect somewhere else.

**The repair.** `calibrate_bar` gains `standardise: bool = True`; the control
optimises `(y − mu) / sigma` and is scored as `net(feats) * sigma + mu` against
raw `y`, exactly as `run_arm` does.

**GREEN: 5 passed in 8.04s.**

| reading | before | after |
|---|---|---|
| `trained_two_feature`, e2, `steps=150` | `2.446645` — BAR BROKEN | `0.699440` — BAR CALIBRATED |
| scale-invariance gap at `c = 0.1` | `20.826173` | `0.000000` |
| scale-invariance gap at `c = 10.0` | — | `0.000000` |
| `negation_scope` at `steps=150` | — | `0.046744`, CALIBRATED |
| clauses 1-4 | — | bit-identical, compared with `==` |

**One existing test had to change, and I caused it.** Baseline
`tests/cameron/test_m3_etasks.py` was **34 passed in 6.09s** on the stashed tree;
after the repair, **1 failed, 33 passed**, the failure being
`test_the_consequence_bar_is_broken_at_the_shipped_step_budget`, which asserted
`not ok150` — i.e. it regression-locked the defect and named it in its own
docstring. That is not a pre-existing failure I am forbidden to touch; it is one
my change caused, and deleting the assertion would delete the finding. It now
holds **both** readings at the same 150 steps on the same batch: the broken one
via `calibrate_bar(standardise=False)`, the repaired one on the default, plus
the bit-identity of clauses 1-4 and the 600-step reading `E2_STEPS` was chosen
to work around. Back to **34 passed**.

That is the only reason `standardise` exists as a parameter. A defect that can
only be described and not re-measured stops being a finding and becomes a story,
and the vacuity rule requires a repair to be shown to change the object it
repairs.

**I did not train `e2_consequence`.** That is Mercury's execution job. What
changed is that its bar now calibrates at the shipped budget, so the run is
possible.

---

## 5. A FINDINGS correction, stated loudly as the contract requires

**FINDINGS B4 is partly wrong.** It says `scale/arm_s.py:55-58` *cites*
`gram_audit` and that this was *"recorded as doc rot, never fixed"*. The file at
`74e5590` reads, at `:56-58`:

> no function or test named `gram_audit` exists in this repository, and an
> earlier revision of this sentence claimed one did (doc rot, recorded as
> STATE.md item 39)

The retraction is already in place, in the file, and it goes on to name which
real artefacts are the nearest analogies (`positivity_audit`,
`scale/foreman_hilbert.py:305-334`, 16861/523776 zero entries at s=1024) and
what they do not cover. What survives of B4 is the true and narrower claim that
**no measurement of pivot-coordinate underflow is on record**. `MISTAKES.md`
P-5 states it that way and flags the FINDINGS entry.

Two further line-reference drifts inside FINDINGS itself, consistent with its
own B7: A4 cites `scale/impact.py:26` for the gate clause, which is at `:23`;
A1 cites `make_impact_hetero_batch` at `scale/impact.py:594`, which is `:601`.

---

## 6. Test summary

| file | result |
|---|---|
| `tests/cameron/test_impact_hetero_is_not_its_own_baseline.py` | 6 passed (RED 3F/2P first) |
| `tests/cameron/test_bar_control_sees_the_arms_preprocessing.py` | 5 passed (RED 4F/1P first) |
| `tests/cameron/test_m3_etasks.py` | 34 passed (34 before, 34 after) |
| `tests/cameron/test_m3_counter_squared.py` + `test_u1_rag_registration.py` | 44 passed |
| `tests/cameron/test_harmonic_attribution.py` | **11 failed, pre-existing, untouched** |

45 passed across my three files. The full suite was never run.

---

## 7. What I could not validate

The `impact_hetero` repair is bound at the registry and the oracle level and
nowhere else: no arm has been trained on the heterogeneous corpus, so I have
shown that the two keys now draw different tensors and that each oracle
reproduces its own label, not that the heterogeneous task is *learnable*,
*harder*, or *worth registering*. FINDINGS A4 stands untouched —
`scale/impact.py`'s own docstring at `:23` forbids registration before four
gates and planted controls, and those four gates have still never executed for
either key, so both entries remain registered against their author's stated
condition and my repair makes `impact_hetero` a correct instance of a task that
has not earned admission. I did not verify the two-block plant is heterogeneous
in the way X21 intends: I checked the corpora differ and that the degree
structure differs enough to move the query node from 673 to 866, not that the
degree distribution has the two-block shape `build_heterogeneous_graph` claims —
that is a distributional claim and I measured a tensor gap. On A7, I established
that the control's reading no longer depends on the label's units and that the
e2 bar now calibrates at 150 steps, but I did not establish that `0.699440` is
the *right* number: `E2_STEPS = 600` was chosen under the defect and nobody has
re-derived the budget e2 actually needs now that the control and the arms agree,
so the step budget in the tree is a workaround whose premise I removed. The
non-e2 tasks that pass through `calibrate_bar` were checked only through
`test_m3_counter_squared.py` and `test_u1_rag_registration.py` (44 passed) and
through the `negation_scope` default; I did not sweep every registered task's
`trained_two_feature` before and after, so a task whose label scale happens to
sit near 1.0 could have moved by an amount I have not looked at, though the gate
is a threshold at 1.0 rather than an exact value and standardisation can only
reduce the reading. The 11 `test_harmonic_attribution.py` failures I attribute
to a missing `absorbing_boundary_kernel` on the strength of the AttributeError
and a grep showing zero occurrences at `74e5590`; I did not investigate whether
that module was meant to exist, which is a live FINDINGS-B2-shaped question I am
leaving open. Finally, `MISTAKES.md` entries V-2 through V-12 and all of P and M
rest on documents rather than on re-running the experiments they record: I
verified every citation's location and quoted text by direct read, and spot-read
six of them in full context, but I did not reproduce a single one of those
fourteen strikes, so their *numbers* are `READ`, not `RUN`.
