# workdoneplanetrum

Round R9, branch `feat/r9-causal-consequence`, from `74e5590`.
Six Opus planets carrying one Baker Street role each, with haiku/sonnet moons.

**100 commits. 35 new test files. 6 new `scale/` modules. 3 new root documents.**
**No published number was moved.**

---

## 1. What the round was for

A handwritten napkin note asked for attention that reads **action → consequence**
rather than nouns → retrieval: *"I went to New York"* should yield *"how was your
travel"*, not *"New York is nice."*

Eight rounds had tried to measure that and the deciding number kept reading zero.
This round set out to find why, build the corpus that could answer it, and run the
measurement.

It did all three. The answer is not the one the lane was built to show.

---

## 2. THE ANSWER — the C1 deciding measurement, complete at 4/4 rungs

`c1_propagate_t{1,2,8,32}`, label `[n, s−t*]`, `s=64, d=24, n_train=2048`,
5 seeds, `softmax` + `twinrow` + `settledrow`.

| rung | `softmax` | `twinrow` | `settledrow` | clears bar | `settledrow − twinrow` |
|---|---|---|---|---|---|
| `t*=32` | 1.003157 | 1.002587 | 1.003214 | none | **void (row G)** |
| `t*=8` | 1.000933 | 1.000774 | 1.001047 | none | **void (row G)** |
| `t*=2` | 0.996745 | 0.999082 | 1.000150 | softmax, twinrow | **void (row G)** |
| `t*=1` | **0.991038** | **0.997688** | **0.999064** | **all three** | **−0.001376, n+ 0/5** |

**Three facts, and they must travel together.**

1. **Where the theory predicts hardest — `t*` at or above the hop budget of 2 —
   every arm fails predict-the-mean outright.** Row G voids every contrast on
   three of four rungs. Nothing learns, including plain softmax.
2. **The only creditable deciding contrast is at `t*=1`, the rung the theory
   predicts least on**, and there **settling loses to its own unsettled twin,
   unanimously**: `−0.001376`, CI `[−0.002031, −0.000776]`, `n+ 0/5`.
3. **`softmax` beats both ceq arms at `t*=1`, unanimously, by about 5× that
   margin.**

**This is not an underpowered null.** Realised `sd_paired` ran `0.000233`–`0.000841`
across rungs; **one seed sufficed at every rung**.

**How to read it:** a **negative result on the ceq arm** in the region where the
corpus is learnable, and **no result at all** in the region the theory is about.
Reporting only the second would hide a real loss. Reporting only the first would
claim the theory was tested where it was not.

The corpus is correctly built (gates pass, closed-form truncation law with no
fitted constant, `k = t*+1` bitwise exact, do()-bit movement test fires). The arm
reaches **88.1–100 %** of the label's support against the shipped arm's
**1.6–3.1 %**. The instrument resolves to one seed. **And the region where `t*`
exceeds the hop budget is unlearnable by every arm at 150 steps and
`n_train=2048`.** That last clause is the round's open question, not its
conclusion.

### The pre-registered counter-prediction scored 4/4, both ways

Venus filed, before any C1 number existed, that the settling contrast would read
inside `±0.027260` at every rung with every CI covering zero.

- **Magnitude half: TRUE at every rung** — `0.000627` / `0.000272` / `0.001068` /
  `0.001376`.
- **Covers-zero half: FALSE at every rung.**

The precise reading: **the settling contrast does move, detectably and
unanimously — by one to two orders less than the resolution anyone pre-registered
as meaningful.** *"Does not move"* is false at the resolution achieved; *"moves
less than `RESOLUTION_13`"* is true.

She split the prediction into two independent booleans before any data existed.
That choice is the only reason this reads as a finding rather than a grade.

---

## 3. The other measurement: it is the trained mixture, not the mixture

The README claimed *"the gain is the mixture, not the equilibrium"*, resting on
the one-hot control reading `−0.118456` worse than softmax.

Mars filed that this confounded *mixture vs lookup* with *trained vs untrained
selection* — `argmax`'s alpha carries **no `grad_fn`**, measured gate gradient
`0.0` against `52.32` (twin) and `55.15` (settled).

He then built `argmaxste`: forward **bitwise identical** to `argmax`
(`torch.equal`, 3/3), backward the softmax Jacobian, `n_params == 4769` unchanged.

| cell | mean NRMSE |
|---|---|
| `argmax` (untrained selection) | 1.010779 — worse than predict-the-mean |
| `softmax` | 0.892323 |
| **`argmaxste`** (one-hot, **trained**) | **0.785019** |
| `twin` (full 8-row mixture) | 0.780927 |

**A one-hot lookup with a trained readout matches an eight-row mixture.** The
mixture half of the claim was withdrawn — in **four** places, not the two anyone
had found. The equilibrium half stands untouched: no `settled` cell was in that
reading.

**Jupiter then sharpened it further, and corrected the controller's framing.**
There are two stages:

- **Stage A** — *which* `k` pivots, `topk(key.norm)`. **Never trained in any arm**,
  byte-identical across every cell measured this round.
- **Stage B** — the mixture over the chosen `k`. This is where the STE lives.

So the finding is **the trained mixture over a fixed selection**, not trained
selection. The controller's dispatch had pointed the maths survey at stage-A
machinery — one stage away from the measured result.

---

## 4. Defects found and fixed

### Live in the tree at round start

| # | Defect | Fix |
|---|---|---|
| 1 | **`impact_hetero` was a vacuous control** — registered `make_impact_batch`, so a heterogeneous-plant arm drew the homogeneous corpus and could not differ from its baseline | Plant bit rides in the tensor as `CH_HET` behind one shared `_graph_from_x`. Discriminates **8/8** draws. The prescribed one-line rebinding was **rejected by Saturn as worse** — all five consumers hardcode `heterogeneous=False`, so it would have swapped a vacuous control for a mislabelled one |
| 2 | **`e2_consequence` bar broken at `2.446646`** — `calibrate_bar` trained its control on raw `y` while `run_arm` trained arms on standardised `y` | `0.699440` CALIBRATED at the shipped 150 steps; scale-invariance gap `20.826173 → 0.000000` |
| 3 | **`run_bucket` counted the whole journal**, so `remaining` went negative | Counts the intersection with the requested units |
| 4 | **`E_T_STAR` KeyError** on three tasks | Guarded, no dial invented — a wrong difficulty dial reaches the pre-registered table |
| 5 | **`Eprocess.value` overflowed inside its own designed range** — crashed at draw **10135 of 10240**, and **the must-fire battery never exercised the class production reads live data with** | Log-space primitives following the `log10_max_attainable` precedent; every representable value bit-identical. `calibrate` now replays the shipped class and refuses at `1e-9`, shown FALSE under a drifted `update` |
| 6 | **A pre-registration whose producer never existed** — `test_harmonic_attribution.py` cited nine symbols on `negation_scope` absent from **every ref in history**, with pilot numbers nothing could have produced | Struck per the `5.4944e-13` precedent. 302 lines kept verbatim, module-level `skip` not `xfail(strict)` so these eleven are not filed alongside the ~50 that *are* the record. 11 failed → 1 skipped |
| 7 | **The struck-constant scanner scanned nothing** — exclusions tested against absolute `p.parts` while every worktree sits under `.claude/worktrees/`. **366 candidates → 0 survivors, exit code 0** | Relative filter; `main()` returns 1 on an empty scan; must-fire plants the constant on disk and requires selection to reach it |
| 8 | **A verdict function that named the wrong cell** — `verdict_of` printed `SETTLED WINS`/`TWIN WINS` for any contrast | Collapsed to one implementation with optional names, defaults pinned. **This landed one turn before the first creditable C1 row, which under the old version rendered `TWIN WINS` when softmax won** |
| 9 | **Partial-ladder verdicts** — rows F (a theory-death verdict) and H fired with no `complete` guard and sat *before* the guard that would have caught them; row C printed a universal over every rung while scanning two | Guards added, 3/6 RED → 6/6 |
| 10 | **A G2 trap**: a hardening test pinned `"slope": -1.389` while the shipped package holds `None`/WITHDRAWN. Repairing the obvious way would have **re-shipped a struck constant** | Flagged, repair direction named: fix the assertion, not the data |
| 11 | **A tolerance band centred on a withdrawn number** — `PUBLISHED_SLOPE = -1.389 ± 0.35` would reject a *better* re-measurement as a regression | Flagged for its owner |
| 12 | **`IMPACT` was unusable three independent ways** — unlearnable across its own split (train graph at `seed`, eval at `seed+12345`, `A`/`B` never in `x`), trivially linear within a graph, and needs `s=1024` where one activation is 8192 MiB against an 8188 MiB card | Retired as this round's corpus; the repairs stand, making it a correct instance of a task that has not earned admission |

### Mechanized prevention (CEQ v13)

- **X-R1 identity manifests** — content hash of `(code path, config, tensor shapes,
  RNG plan)`. Must-fire fires on five config fields, **each naming the field that
  moved**; restore reproduces the hash. **G2 proven bitwise on a published cell.**
  It draws the journal-key collision rather than arguing it: three of five
  identity-defining arguments collapse to one key, so the second run silently
  overwrites the first.
- **Kirchhoff dual oracle** — matrix-tree cofactor as an independent computation of
  the same `ω_v`; both run per instance, agreeing to `2.2e-16`–`9.6e-14`. A planted
  off-by-one moves `ω` by `0.175`–`0.316` and is **caught 6/6**.
- **`n+` seed-agreement printed beside every interval**, applied at the shared
  function so **every reading in the repo carries it, including the headline**.
- **Exact interval pairs and atom counts** printed beside every bootstrap interval,
  with a permanent `n_atoms ≤ C(2n−1, n)` guard.
- **A union merge driver** for the append-only pytest event log, which was
  conflicting on every parallel merge.

---

## 5. Corrections to the author's own work

| What was claimed | What is true |
|---|---|
| v11.1: random role coherence `≈ 0.147` at `d=256, k=16` | **`0.174795`** — MC CI `[0.174460, 0.175131]`, corroborated `0.174499` by order-statistic quadrature. **15.8 % low**, because the union runs over `C(k,2)=120` pairs, not `k`. M5's conclusion *strengthens* |
| `LOOP_PROMPT.md:53-55`: representation-iteration is *"the one live direction not yet measured"* | Stale. `foreman_looped.py` is a complete instrument with binds and a pre-registration; the journal held 2 of 6 cells (now 4) |
| `BOARD.md:288`: U2/U3 scaffolding *"exists as stubs"* | False. Zero `.py` anywhere — prose only |
| `PIVOT_EXCLUSION_FALSIFIER.md:156`: `settled_plus` *"still running"* | It finished, at `0.956787` — also worse. Never updated |
| `capability_table`: *"`m3_quintuple` has no `--task` flag"*, and *"carries NO trained weights"* | Both false since `--task` landed and `0162bdd` shipped a manifest. **The generator would have clobbered that manifest on the next table cut** — a second clobber was still armed behind a documented hand-edit workaround |
| The published intervals appeared to split into two estimator families | **No number is wrong.** There is **one discrete statistic with 126 atoms**, and the bootstrap lands on a neighbouring atom depending on seed. Every "family" observation this round reduces to that |
| `test_harmonic_attribution.py`'s pre-registered clause | No producer ever existed, in any ref |

---

## 6. Corrections to the controller's own work

The author asked to be outdone. So was the controller, repeatedly, and this is the
honest list.

| What I said | What was true | Who caught it |
|---|---|---|
| *"Softmax is **provably** Bayes-optimal on that label shape"* — **the round's founding premise** | Overstated three ways: the paper's predictor is `erf` not softmax; optimality is asymptotic under `L = o(d)` while this repo runs `L/d = 4.00`, **the opposite direction**; Prop 3 refutes linear *regression*, not linear attention. **The label-shape worry survives; "provably" does not** | Jupiter |
| *"Drop the `[:, s-1]` index — one index change"* | Necessary but **not sufficient**. Measured on 256 instances: the naive vector readout is **bitwise identical across all five cells** at positions `0..s-2`; 98.4 % of gradient would carry no cell information | Venus predicted, Neptune measured |
| *"The label is `[n, s]`"* | `[n, s−t*]`. A strictly causal operator raised to `t*` vanishes on the first `t*` coordinates, so `[n, s]` would ship `t*` positions of `sd 0` — the fourteenth strike in a new shape | Saturn |
| *"`t*=1` is UNDERPOWERED, 62 seeds needed"* | **Wrong, and three of us verified the arithmetic without checking the input spread belonged to the lane.** The `0.109199` is the **e3 scalar** ladder's spread. C1's is `0.000841` — **130× smaller**. Every rung needed **one** seed. `t*=1` is the *best*-resolved rung | Mercury |
| *"The `verdict_of` change risks invalidating journalled units"* | The hazard does not exist as stated. `contrast` has **zero call sites inside `_unit`**, and **no journal holds a `verdict` key at all** | Mercury |
| *"`e_ladder` and `page_trend` both call `contrast`"* | `page_trend` never calls it — docstring prose | Mercury |
| *"128 atoms and 126 atoms are different lattices"* | I **amplified a rationalisation**. 126 is the generic count; 128 was a float-associativity bug. A correction built on a wrong explanation is worse than the original error | Mercury, on himself first |
| *"+3hop clears `FAIL_BAR` by `0.0049`"* | It clears by `0.0951`. `0.0049` is its distance **below the mean predictor** — the sharper point: all four decoders required to fail sit within `0.0049` of a constant | Mars |
| FINDINGS section B, generally | Falsified in **six** places across four planets. Assembled from agent output the controller never line-verified. Demoted to unreliable as a whole | Mars, Mercury, Saturn, Neptune |

---

## 7. What each planet delivered

- **Saturn (WATSON)** — `MISTAKES.md` (36 entries, four families), the `c1_propagate`
  corpus, X-R1 identity manifests, the `CH_HET` control repair, the `e2` bar fix,
  a citation binder for the file that documents citation failures. **Refused to
  pad a mistake type to three sourced instances when he could evidence two.**
- **Mercury (LESTRADE)** — the entire C1 deciding measurement, the STE reading, the
  e-process overflow, four mechanical fixes, measured pricing that corrected
  `STATE.md` by 3.0×, the table cut, three hand-offs. **Retracted his own number
  and his own rationalisation for it.**
- **Mars (MORIARTY)** — 15 attacks filed, **12 fired**. The `argmaxste` cell that
  settled his own GREEN attack. The **thirteenth defect class**. Proof that M4's
  rejection region is **empty, not unexercised** — a NaN in the crushed slot leaves
  output bitwise identical 40/40. **Discarded his own first statistic for failing
  his own vacuity rule.**
- **Venus (IRENE)** — the pre-registration hole audit (8 fall-throughs, 4 code
  drifts, three now fixed and RUN-class), the struck harmonic clause, the re-audit
  prediction **filed in a separate commit before touching adjudication**, and the
  C1 counter-prediction that scored 4/4. **Ruled the struck-constant boundary
  against her own interest.**
- **Neptune (LINUS)** — the systems gate that killed the naive plan on 256
  measured instances, the per-row arm, the coverage number that justifies the lane,
  the withdrawal of the refuted claim in all four places, capability-table truth
  repair. **Corrected his own cost gate by 5×.**
- **Jupiter (MYCROFT)** — the Page lattice proof (51 achievable p-values, critical
  `L=137`, exact size `0.037002877`), the Kirchhoff dual oracle, the RIP line, the
  corrected coherence constant, and the 16-candidate maths survey. **Labelled 10 of
  16 candidates moon-class and un-re-verified rather than averaging them in.**
- **Moons** — Titan (verdict guards), Deimos (8 attacks, 8 fired, including the
  e-process crash and a pre-registration with no producer).

---

## 8. Open questions

1. **Is the `t* > hop budget` region unlearnable, or just under-trained?** Every
   arm fails predict-the-mean there at 150 steps and `n_train=2048`. A capacity or
   step sweep separates "the task is too hard" from "the budget is too small". This
   is the round's central open question.
2. **Row G is a binary gate at exactly `1.0`.** At `t*=2`, `softmax` clears it by
   `0.003255` and `twinrow` by `0.000918`, and their contrast is creditable *by the
   letter*. Whether a gate asking "did anything beat the mean" rather than "by
   enough to matter" is doing its job is a live question. **Deliberately not acted
   on** — refitting a pre-registered threshold after seeing data is the defect
   `RESOLUTION_13` is frozen against.
3. **At `N=5` the finest achievable two-sided p is `0.0625`, not `0.05`.** The sign
   lattice has **zero** two-sided p below `0.05`; Page's L escapes at exact size
   `0.037002877`. Every 5/5 claim in this repository inherits this.
4. **E4′'s strike hangs on one unrepeatable draw.** Both seeds hardcoded, spread
   never measured; 1 of 20 draw seeds reverses it, and the shipped draw sits
   `0.78 sd` below the mean of the other twenty, on the side that licenses the
   strike.
5. **The re-audit has not been run.** Venus has filed **1 of 5** GREENs surviving —
   or **0 of 5** if the `N=5` floor counts as a fifth gate, and she asked for that
   scoring convention to be made visible rather than silently chosen. Three of the
   four she downgraded now have a fired adversarial attack behind them.
6. **Untouched**: `scale/recall_probe.py` states its own identity, names the
   non-artefact quantity, has zero importers and no results. Jupiter: *"already
   sitting unrun."*
7. **`ceq/nash.py`'s shared-`tau`** is real but bounded too small to be the cause —
   median attenuation `0.789701`, and a `1.27×` shrink cannot produce OOD NRMSE
   `2.6151`–`5.8198`. Downgraded from a rerun to a one-line check.

---

## 9. The process finding

**Value comparisons went 0-wrong. Structure comparisons went 9-for-9 wrong.**

Every correction in section 5 and 6 above was caught by someone measuring a number,
and almost every defect in section 4 was created by someone comparing a name, a
shape, a string, or a fixed list instead.

The thirteenth class Mars found is the sharpest instance: **the control constructs
its own input, so it certifies the instrument's predicate over a domain production
never chose.** Three instruments this round passed their own real, planted,
non-degenerate controls while reaching nothing — an e-process battery testing a
class production never uses, a scanner whose control injected text past target
selection, and an interval binder parameterised over a fixed list rather than
scanning the documents it polices.

All three satisfied the existing vacuity rule. The defect sits one level up:
**the control validates the instrument's mechanism and never its scope.**
