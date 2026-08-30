# MARS / MORIARTY — R9 iteration 1

Worktree `.claude/worktrees/agent-afb2a8ed4e030ff6a`, reset onto
`feat/r9-causal-consequence` @ `74e5590` (it had been eight commits behind at
`ac47049`, where `scale/impact.py` does not exist).

Commit: `3180d34` — `tests/mars/test_mars_r9_iteration1.py`, 10/10 pass in
51.83 s. No existing test touched, no existing test run.

**Attacks filed: 10. Fired: 8. Refuted by my own check before filing: 4
(listed in §Refuted).**

---

## The correction first — FINDINGS B5 is wrong

`FINDINGS.md` B5 says of the IMPACT gate numbers in commit `74e5590`'s body:
*"Nothing on disk reproduces them. No results file, no test, no CLI."*

The second sentence is right. The first is wrong, and it matters because it told
every later agent the numbers were invented. `impact_report(n=512, s=1024,
seed=0)` reproduces all five to the printed precision, in 58 s:

| claim in the commit body | measured this session |
|---|---|
| decoder local `1.0039` | `1.0039067318125765` |
| decoder planted `0.06738` | `0.06738566822969232` |
| truncation k2 `0.3243` | `0.3243630511140254` |
| sign degrade `0.8793` | `0.879385727796889` |
| sign CI `[0.8300, 0.9145]` | `[0.8300754113928057, 0.9145472963019973]` |

Class `RUN`. All three executable gates return `passes: True` / `fires: True`.
The defect was never that the numbers were fabricated. It is that the gates they
came from do not test what they say they test — §M2, §M5, §M6, §M7 below — and
that the task they gate cannot be learned at all, which is §M1.

One cause of the confusion is worth naming: `impact_report` cannot be printed on
a Windows cp1252 console. It builds the whole report, then dies on the `→` in
its own title line with `UnicodeEncodeError`. Class `RUN`.

---

## Ranked attacks

### M1 — IMPACT is trivially linear within a graph and unlearnable across the split the harness actually draws. `CONFIRMED`, three independent paths

Cost if true: the entire U4a T-FAMILY 2 lane. Every arm must read ≥ 1.0 and the
between-arm contrast is noise.

**Mechanism.** `scale/m3_quintuple.py:478-480` draws train at `seed` and eval at
`seed + 12345` — the repo-wide held-out convention, present in nine harness
files (`m3_capability.py:281,308`, `etask_k5e.py:65`, `paired_arm.py:55`,
`foreman_looped.py:195`, `m3_synthetic_settled.py:214,232`,
`fgreen_matched.py:163`, `eprocess_perdraw.py:109`, `foreman_consequence.py:427`).
For every chain task the structure lives in the input, so a different seed is a
proper held-out draw. For IMPACT `scale/impact.py:566` sets
`graph_seed = int(seed % 1000000)` and builds the graph **outside** the example
loop, so:

- one graph carries all `n_train` examples;
- the eval batch sits on a *different* graph;
- the label is `y = w · news` with `w = Bᵀ Kᵀ e_q` fixed per graph
  (`scale/impact.py:580-583`);
- neither `A` nor `B` enters `x`. Channels 2..9 are declared *"reserved for
  sparse B encoding if needed (not used)"* at `scale/impact.py:69`. `x` carries
  news, degree, `rho`, and the graph seed as a float.

**Path A — closed form, no fitting.** `w₀` for graph 0, `w₁` for graph 12345:

```
query node train 673, eval 564
cos(w0, w1)                        = -1.0769638600689055e-08
transfer NRMSE, best rescaled      = 0.9999999999999999
transfer NRMSE, unscaled           = 1.4341072133181152   (analytic sqrt(2))
```

**Path B — regression, most generous linear probe available to any arm**
(all 1024 news + all 1024 degrees + intercept, `n_train=4096`, `n_eval=4096`):

```
planted positive, train-graph held out   NRMSE = 4.926712551941536e-08
TRANSFER          eval-graph             NRMSE = 1.4782066277450256
planted positive, eval-graph held out    NRMSE = 5.195106050150856e-08
mean predictor    eval-graph             NRMSE = 1.0002929510434602
```

**Path C — label correlation, no model at all.** Same `news` vectors, two
graphs: Pearson `r = 0.007232892657207252`.

The three fail differently: A is linear algebra on the planted matrices, B is a
least-squares fit on drawn tensors, C is a correlation of two label vectors.

**Two findings, not one.** The planted positive at `4.93e-08` is itself an
attack: `make_impact_batch`'s docstring says *"cross-asset propagation
required"* (`scale/impact.py:632`). It is not. Within a graph the task is a fixed
linear functional of channel 0, solvable exactly by a linear readout with no
attention, no propagation, no iteration, and no knowledge of the graph.

**Falsification attempted.** Could an arm use `CH_SEED`? It is one scalar
constant per batch; recovering a 1024×1024 MP-cleaned Rips-backbone graph and a
1024×1024 signed plant from it is not a function any arm here can express, and
the eval seed is never seen at train time. Could `n_train=8192` supply many
graphs? No — the `build_impact_graph` call sits outside the `for idx in range(n)`
loop, so all 8192 examples sit on one graph. Both refutations fail.

**Downstream.** This makes `FINDINGS.md` A1 (`impact_hetero` registers
`make_impact_batch`, re-confirmed `RUN`: `M3_TASKS['impact_hetero'][0].__name__`
is `make_impact_batch`) moot rather than merely vacuous — both keys are
unlearnable. It also means the task is unreachable at the harness default: at
`s=64` the builder raises `ValueError: s=64 nodes is below IMPACT admissibility
floor 1024`, and `E_T_STAR` contains neither `impact` nor `impact_hetero`
(`RUN`, re-confirming A3).

### M2 — Gate 3 fires on a perturbation with zero sign flips, and its own must-fire column reads negative. `CONFIRMED`

Cost if true: the sign gate, one of four admission conditions, does not test the
claim the corpus definition rests on (*"planted sign structure IS ground truth"*,
`scale/impact.py:5-7`).

**Mechanism.** `impact_sign_gate` fits `X_plant_aware` — features built from the
**planted** `B` — against a label rebuilt from a **perturbed** `B`. Any
perturbation drives the features off the label. The PASS condition at `:948` is
`degrade_aware > 0.10 and ci_lo > 0.0`.

Controls run on identical instances, identical features, identical split:

| perturbation of `B` | sign flips | `degrade_aware` |
|---|---|---|
| (a) null — `B` unchanged | 0 | `5.394032692729311e-09` |
| (b) shipped sign scramble | ~2048 | `0.879385727796889` |
| (c) magnitudes redrawn from the same U[0.5,1], **signs kept** | **0** | `0.18305939760911277` |
| (d) column permutation, each row's sign multiset kept | 0 net | `0.9304273176972595` |

(c) with the gate's own 400-resample bootstrap: CI `[0.1744402354846823,
0.23106461469419337]`, so `degrade > 0.10 and ci_lo > 0` is **True**. The gate
passes on a `B` whose every sign is the planted sign.

**Adversarial pass — and it costs me part of the claim.** At seed 1 (c) reads
`0.070133`, **below** the bar. At seed 2 it reads `0.145173`, above. So the
specificity failure is **2 of the 3 seeds measured**, not all three. The null (a)
fires correctly at every seed, so the gate is not trivially broken — it is
under-specified.

**Second half.** `degrade_local = -0.0011102202196815458` — negative. The
docstring at `:872` says *"sign-scrambled B must degrade **every arm's** score"*
and *"Must-fire seen firing"*. The local arm's score improves. `fires` at `:948`
reads only `degrade_aware`, so the half that does not fire is computed, returned
in the dict, printed by `impact_report`, and excluded from the verdict.
Catalogue mechanism: a must-fire control reported and not enforced.

### M3 — the `argmax` control's pivot selection is never trained, so the standing GREEN's mechanism claim is confounded. `CONFIRMED`

**This is the attack on the strongest standing GREEN.**

Cost if true: not the `+0.111396` itself, which stands. What falls is the
*decomposition* — *"the MIXTURE is the whole contribution and the EQUILIBRIUM is
none of it"* (`CHECKLIST.md:1206`, repeated at `DONE.md:795,1363,1725`,
`D1.md:498`).

**Mechanism.** `scale/m3_quintuple.py:295-297`:

```python
alpha = torch.zeros_like(log_gate)
alpha.scatter_(1, log_gate.argmax(dim=-1, keepdim=True), 1.0)
```

`torch.zeros_like` does not require grad and `scatter_` of a constant creates no
grad path. Measured on the identical batch (`negation_scope`, n=64, s=64) at
identical init, probing with a random weight vector:

| cell | `alpha.requires_grad` | `alpha.grad_fn` | `‖∂α/∂log_gate‖₁` |
|---|---|---|---|
| `twin` | True | `ExpBackward0` | `52.3193` |
| `settled` | True | `ExpBackward0` | `55.1514` |
| `argmax` | **False** | **None** | **`0.0`** |

`twin` and `settled` learn *which pivots to weight*. `argmax` inherits
`log_gate.argmax()` from a gate that receives no gradient through this path for
the whole of training.

**What that does to the inference.** `argmax − softmax = −0.118456`, 0/5,
therefore measures softmax against *"read one pivot chosen by an untrained
gate"*, not against *"read the best pivot"*. The gloss *"reading one pivot is
worse than reading none"* is not licensed by it: mixture-vs-lookup is confounded
with trained-vs-untrained selection.

**The number that would settle it.** A hard-selection cell whose selection *is*
trained — straight-through or Gumbel top-1, expressible at the same
`n_params=4769` as a constructor float, so hard rule 3 holds. If it reads at or
below softmax's `0.877168` the mixture claim falls; if it still reads near
`1.010779` the claim survives and is then earned. Not run here.

**Falsification attempted.** Does `argmax` lose gradient overall? No — `wq`/`wk`
still receive gradient through `av` and through the shared softmax operator
(measured end-to-end `‖g wq‖ = 2.146141e-05` for argmax vs `9.478988e-06` for
softmax on the same batch), so the arm is not starved. Only the *selection* path
is dead, and the claim is scoped to that. Note also that `select_pivots` is
non-differentiable for **every** cell (`FINDINGS.md` E), so the asymmetry is
specifically the weighting stage, not the candidate stage.

### M4 — the same `argmax` number is voided for one sign of the comparison and credited for the other. `CONFIRMED` (`READ`)

`ceq/hf_artifact/README.md:18` — `argmax` mean `1.010779`, sd `0.010115`, CI
`[1.008307, 1.042529]`, **"NO — credited with nothing"**. The interval excludes
1.0 from below, so `argmax` is significantly worse than predicting the mean.

`CHECKLIST.md:1207` applies the rule: *"`settled vs argmax = +0.226893` is a win
over a failure and licenses nothing — the largest positive number in the table is
the least meaningful."*

`CHECKLIST.md:1206` then reads the same arm's deficit in the other direction as
positive evidence for a mechanism. A comparison against an uncreditable arm
cannot license a conclusion when read one way and license nothing when read the
other. Independent of M3, and it needs no run.

Per-seed nuance, stated because it cuts the other way: `argmax` reads `0.994867`
at seed 1, below 1.0. The "credited with nothing" verdict is on the mean, not on
5/5.

### M5 — Gate 4 has no pass condition, and one of its per-instance covariates is a construction constant. `CONFIRMED`

`impact_report` writes Gate 4's verdict as a literal string —
`w(f"  => PASS (exact eig, Gromov four-point, branching ratio)")` — with no
condition evaluated, and `all_pass = status1=="PASS" and status2=="PASS" and
status3=="PASS"` omits it entirely. The module docstring's *"all four gates"*
(`scale/impact.py:12`) is three gates and a printed string. Vacuity rule 7: a
gate that structurally cannot fail.

The covariate it prints is partly a constant of construction.
`scale/impact.py:311` sets `rho = RHO_TARGET / spectral_radius_raw`, and
`impact_covariates` reports `rho * spectral_radius_raw`:

| seed | `spectral_radius` | `lambda2` | `hawkes_branching` | `delta_hat` |
|---|---|---|---|---|
| 0 | `0.88` | 0.849944 | 0.805514 | 1.0 |
| 1 | `0.88` | 0.873865 | 0.805412 | 0.5 |
| 2 | `0.88` | 0.824605 | 0.804958 | 1.0 |
| 12345 | `0.88` | 0.823851 | 0.806050 | 0.5 |

`spectral_radius` is `RHO_TARGET` identically at every seed. `hawkes_branching =
0.88*0.90 + 0.1*raw_branch`, so `0.792` is **98.32%** of its value at seed 0 and
it ranges over `1.09e-03` across four seeds. `delta_hat` takes two distinct
values across four seeds. Only `lambda2` varies meaningfully. The commit title —
*"per-instance lambda2/deltaHat covariates"* — is right about `lambda2` and
overstated for the rest.

### M6 — Gate 1's FAIL probe is rank 2 advertised as rank 4. `CONFIRMED`

`impact_features` (`scale/impact.py:697-712`) returns
`[1, news_q, deg_q, news_q*deg_q]`. The graph is fixed for the whole batch, so
`deg_q` is one repeated scalar:

```
shape (512, 4)   matrix_rank 2   cond 1409459955957760.0
col2 (deg_q) unique values: [2.2898128]
col3 == col1 * deg_q exactly: True
NRMSE with [1, news_q] only : 1.0039067322353783
NRMSE with all four columns : 1.0039067318125765
```

The two "degree" features contribute `4.2e-10` of the gate's headline number, and
the `1e-6` ridge at `:786` is what makes the singular normal matrix solvable at
all. The gate's conclusion (local probe fails) is unchanged; its stated content —
that degree was available to the local probe and did not help — is not measured,
because degree is a constant column.

### M7 — the rule-4 non-degeneracy guard is an identity. `CONFIRMED`

`impact_decoder_gate:816` and `impact_sign_gate:922` install
`0.0 < frac_median < 1.0` as the guard against the fourteenth strike (a PASS case
whose label was constant). `frac_median` is the fraction of a sample strictly
above **its own** median:

```
gate label frac                  = 0.5
pure N(0,1) noise, n=512         = 0.5
constant 3.0 + 1e-12 noise       = 0.5
label scaled by 1e-30 (sd = 0.0) = 0.5
```

It reads 0.5 for any even-sized continuous sample, including one whose sd is
exactly `0.0`. **Fair scoping:** the same boolean also conjoins
`label_sd > 1e-6`, and *that* conjunct does catch the degenerate case. So the
gate is not broken; the `frac` half is decorative, and `impact_report` prints
`frac=0.5000` forever as if it were evidence.

### M8 — `ceq/rips.py`'s promised independent derivation has no referent. `CONFIRMED` (`READ`), and the counts survive it anyway

`ceq/rips.py:28-30`: *"The counts are therefore checked against an independent
derivation rather than assumed to transfer."*

`_count` (`:154`) calls `components` (`:110`). There is exactly one component
routine in the repo, and every assertion in `tests/cameron/test_e4_rips_gate.py`
compares against numbers that same routine produced. `BASE_PROMPT.md`: rerunning
the same expression twice is not a second path.

One is supplied in the committed test file. Union-find agrees 6/6, and the
floating-point concern the header raises is measured rather than assumed:

| case | ported | union-find | min &#124;dot − cos r&#124; |
|---|---|---|---|
| StableSparse_S2Rips_64 | 15 | 15 | 1.302439e-03 |
| CriticalBridge_S2Rips_256 | 6 | 6 | 1.991397e-05 |
| SupercriticalDense_S2Rips_256 | 1 | 1 | 1.207727e-05 |
| GroundedStaticRepeated_S2Rips_256 | 1 | 1 | 3.067545e-05 |
| StableRepeated_S2Rips_1024 | 178 | 178 | 7.163961e-07 |
| CriticalLarge_S2Rips_1024 | 3 | 3 | 1.583093e-06 |

Worst margin `7.16e-07`, nine orders above the `~1e-16` a 1-ULP libm
disagreement could move a dot product. The counts do transfer. The defect was the
missing derivation, not the fidelity — a doc claim that had no backing until now.

### M9 — three small live defects, grouped

- **`impact_report` is unprintable on cp1252.** `UnicodeEncodeError` on `→` in
  its own title. It builds the entire report (58 s of gates) and then dies at the
  first `print`. `RUN`.
- **`impact_attribution`'s bootstrap silently drops degenerate resamples.**
  `scale/impact.py:1042` filters `np.isfinite(r)`. The planted query row has 4
  nonzeros in 1024, so all-zero resamples give an undefined Spearman: **5 of 400
  dropped**, CI computed over 395. Small (1.25%) and one-directional, but
  undeclared.
- **`QuintArm.__init__` accepts a `kind` positional it discards.**
  `scale/m3_quintuple.py:268-270` — `super().__init__("softmax", s)` hardcodes the
  kind; the real selector is a keyword-only `cell` defaulting to `"softmax"`.
  `QuintArm("twin", 64)` builds a softmax arm with no error. It cost me one wrong
  measurement this session before I caught it. Every in-tree caller passes `cell=`
  correctly (7 sites checked), so this is latent, not live.

---

## Refuted — attacks I killed before filing

- **`lambda2` is a hardcoded multiple of the spectral radius.** The
  `lambda2_raw = spectral_radius_raw * 0.9` line at `scale/impact.py:302` is in an
  `except` fallback only. The main path takes the real second-largest `|eigval|`
  (`:296-299`) and the measured values vary per seed. Dropped.
- **The report's cycle control value is invented.** `impact_report` prints
  *"Cycle n=32 delta_hat = 8.000000"* citing
  `tests/foreman/test_gromov_delta.py:170`, which is parametrised over
  `n ∈ {12, 16, 40}` — n=32 is not tested there, and the n=32 cycle at `:157`
  asserts only `d_cycle > 0.0`. But the value is right: computed
  `gromov_delta(hop_distances(32, C_32)) = 8.0`. A wrong citation on a correct
  number, not worth an attack slot.
- **`_news_for_example` seed collisions.** `(seed*1000003 + idx*9176)` — 1000003
  is prime and does not divide any `9176·Δidx` for `Δidx < 10⁶`, so train and eval
  draws cannot collide. Dropped.
- **`eprocess`'s clip breaks the martingale.** `clip` is `min(x, CLIP_C)`,
  one-sided from above, which makes the e-process smaller. Conservative, not
  unsafe. Dropped.

---

## Claim ledger

| # | claim | class | check |
|---|---|---|---|
| 1 | Gates 1-3 execute and pass; five numbers match the `74e5590` body exactly | `RUN` | `impact_decoder_gate/impact_truncation_gate/impact_sign_gate(n=512,s=1024,seed=0)`, 58.1 s, JSON quoted |
| 2 | FINDINGS B5's "nothing reproduces them" is false | `RUN` | table above, five figures |
| 3 | eval batch sits on a different graph | `READ` | `m3_quintuple.py:478-480` + `impact.py:566` |
| 4 | `A`, `B` are not in `x` | `READ` | `impact.py:69` "reserved … not used"; builder writes only `CH_NEWS`, `CH_A_DEG`, `CH_RHO`, `CH_SEED` |
| 5 | within-graph NRMSE `4.93e-08`, transfer `1.478`, mean `1.000` | `RUN` | linear probe, n=4096/4096, s=1024 |
| 6 | `cos(w₀,w₁) = -1.08e-08`; unscaled transfer `1.434 ≈ √2` | `RUN` | closed form on `Bᵀ Kᵀ e_q` |
| 7 | label correlation across graphs `0.0072` | `RUN` | identical news vectors |
| 8 | sign gate crossed at `0.1831` with 0 flips, CI `[0.1744, 0.2311]` | `RUN` | magnitude-only perturbation, seed 0 |
| 9 | same at seed 2 `0.1452`; **not** at seed 1 `0.0701` | `RUN` | adversarial pass, cost me 1/3 |
| 10 | null control reads `5.39e-09` | `RUN` | `B_scrambled = B` |
| 11 | `degrade_local = -0.0011`, excluded from `fires` | `RUN` + `READ` | gate output; `impact.py:948` |
| 12 | argmax alpha has no grad_fn; gate gradient `0.0` vs `52.32`/`55.15` | `RUN` | `torch.autograd.grad` on identical init/batch |
| 13 | argmax mean `1.010779`, CI `[1.008307, 1.042529]`, "credited with nothing" | `READ` | `ceq/hf_artifact/README.md:18` |
| 14 | the void rule is applied to `settled vs argmax` and not to `argmax vs softmax` | `READ` | `CHECKLIST.md:1206` vs `:1207` |
| 15 | Gate 4's verdict is a literal string, omitted from `all_pass` | `READ` | `impact.py:1120-1124` |
| 16 | `spectral_radius == 0.88` at 4 seeds; hawkes constant share 98.32% | `RUN` | `impact_covariates` × 4 |
| 17 | `impact_features` rank 2 of 4, cond `1.41e15`, Δ headline `4.2e-10` | `RUN` | `np.linalg.matrix_rank` + refit |
| 18 | `frac` reads 0.5 for a label with `sd == 0.0` | `RUN` | three samples incl. degenerate |
| 19 | no independent component derivation exists in tree | `READ` | `_count:154` → `components:110`; grep of tests |
| 20 | union-find agrees 6/6; worst dot margin `7.16e-07` | `RUN` | committed test |
| 21 | `impact_report` raises `UnicodeEncodeError` on cp1252 | `RUN` | traceback captured |
| 22 | attribution bootstrap drops 5/400 resamples | `RUN` | `spearmanr` finite count |
| 23 | `QuintArm("twin", 64)` silently builds softmax | `RUN` | `arm.base_cell == "softmax"` |
| 24 | `impact_hetero` registers `make_impact_batch` (A1 re-confirmed) | `RUN` | `M3_TASKS['impact_hetero'][0].__name__` |
| 25 | `impact` raises at the harness default `s=64`; `E_T_STAR` has neither key | `RUN` | `ValueError` text quoted |
| 26 | a trained-selection argmax would settle M3 | `DERIVED` | **not run** |
| 27 | no arm can learn IMPACT across the split (not just no *linear* probe) | `DERIVED` | argued from claims 3-7 by construction |

---

## What I could not validate

The two loudest claims are not fully closed. M1 is measured with a linear probe,
not a trained arm: claims 5-7 bound exactly what a linear reader can do and rule
out the graph-free linear rule, but the step from there to "no arm can learn it"
is `DERIVED` from the fact that `A` and `B` never enter `x` — I did not train a
single arm on `impact`, and the honest residual is that some non-linear function
of news and degree might carry signal I have not bounded. M3 shows the gradient
path to the pivot gate is absent and that the published contrast is therefore
confounded; it does **not** show what a straight-through argmax would score, so
"the mixture is the whole contribution" is unsupported by the cited evidence
rather than refuted, and could still be true. M2's specificity failure held on 2
of 3 seeds and I ran only three — the honest reading is that the `0.10` bar does
not separate sign from magnitude reliably, not that it never does; I also
bootstrapped only the seed-0 magnitude perturbation, not seed 1 or 2. I did not
audit `scale/e4_harmonic.py`, `scale/e4_harmonic_reroute.py`,
`scale/eprocess_perdraw.py`, or `impact_attribution`'s Spearman path beyond its
bootstrap filter, and my one pass over `ceq/nash.py` found only that
`nash_operator` discards the residual `qre_stance` offers and that `safe_tau` is
enforced when `tau is None` and unchecked when a caller passes one — I found no
call site passing a bad `tau`, so nothing is filed there. The gate numbers in
§Correction are from a single seed at `n=512`; I did not sweep them.
