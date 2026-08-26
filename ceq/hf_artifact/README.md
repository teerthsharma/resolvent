# Capability table v0 -- CEQ signed pivot-routed attention

Task `negation_scope` at `s64_d24_st150_ntr8192_nev512_b21`, 5 seeds, `n_params = 4769` on every arm. Journal `m3_quintuple_v2.jsonl` at commit `9629616` (HEAD `9629616`). Pre-registered reading: `M3_QUINTUPLE_PREREGISTERED_READING.md`. Produced by `scale/m3_quintuple.py --cells softmax glance settled twin argmax --seeds 0 1 2 3 4 --ks 8`.

**Softmax is the baseline and is measured first.** A verdict of NO DIFFERENCE is printed wherever that is what the interval says.

**This package ships no language-model weights, and its table cells carry no weights
either.** The numbers below come from `results/m3_quintuple_v2.jsonl`, which journals
metrics and not tensors. What IS shipped, since the e3 ladder completed, is a separate,
verified set of trained probe-arm tensors under `weights/` -- five 4,769-parameter `twin`
checkpoints at task `e3_t1`, one per seed. See "Weights shipped with this package" below;
none of them loads into `CEQForCausalLM` and none is named `model.safetensors`.

## Arms

| task | arm | eval NRMSE (mean of 5 seeds) | sd | marginal 95% CI, seed 0 | params | beats predict-the-mean | consequence fidelity (1.7c) |
|---|---|---|---|---|---|---|---|
| `negation_scope` | `argmax` | 1.010779 | 0.010115 | [1.008307, 1.042529] | 4769 | **NO -- credited with nothing** | NOT MEASURED |
| `negation_scope` | `glance` | 0.892323 | 0.015866 | [0.830455, 0.924226] | 4769 | yes | NOT MEASURED |
| `negation_scope` | `settled` | 0.783886 | 0.064106 | [0.713106, 0.792690] | 4769 | yes | NOT MEASURED |
| `negation_scope` | `softmax` | 0.892323 | 0.015866 | [0.830455, 0.924226] | 4769 | yes | NOT MEASURED |
| `negation_scope` | `twin` | 0.780927 | 0.016547 | [0.731310, 0.805387] | 4769 | yes | NOT MEASURED |

Per-seed readings, in seed order:

* `argmax`: 1.022910  0.994867  1.011705  1.010804  1.013609
* `glance`: 0.877168  0.889523  0.919148  0.890175  0.885603
* `settled`: 0.753581  0.768802  0.874658  0.816071  0.706317
* `softmax`: 0.877168  0.889523  0.919148  0.890175  0.885603
* `twin`: 0.767403  0.784397  0.794505  0.798001  0.760328

## Contrasts

`delta = NRMSE(reference) - NRMSE(arm)`, so positive means the arm has the lower error. Estimator: paired percentile bootstrap B=10000 seed=0. Strict at zero -- an interval touching zero does not exclude it.

| arm | reference | delta | 95% CI | seeds favouring arm | verdict | note |
|---|---|---|---|---|---|---|
| `settled` | `twin` | -0.002959 | [-0.048587, +0.031557] | 3/5 | **NO DIFFERENCE** | HEADLINE |
| `settled` | `argmax` | +0.226893 | [+0.175040, +0.275862] | 5/5 | **settled WINS** | HEADLINE CAVEAT -- argmax is the attribution control |
| `settled` | `softmax` | +0.108437 | [+0.066232, +0.147110] | 5/5 | **settled WINS** |  |
| `twin` | `softmax` | +0.111396 | [+0.100873, +0.121920] | 5/5 | **twin WINS** |  |
| `argmax` | `softmax` | -0.118456 | [-0.134115, -0.102204] | 0/5 | **softmax WINS** |  |
| `glance` | `softmax` | +0.000000 | [+0.000000, +0.000000] | 0/5 | **NO DIFFERENCE** | G3 binds glance bitwise to softmax at t_max = 0: this row is zero BY CONSTRUCTION, not a measured tie |

## The headline cell is undecided, and that is structural

```
undecided at evidence E_t = 0.9978465225545524 (settled direction), 1.0019156582048507 (twin direction), t = 5 -- STRUCTURAL, not a data outcome: max_attainable(5) = 3.80169140625 < THRESHOLD = 40.0, and MIN_T_MIXTURE = 13, so five seeds cannot cross in either direction whatever the data say.
```

| quantity | value |
|---|---|
| `E_t` settled direction | `0.9978465225545524` |
| `E_t` twin direction | `1.0019156582048507` |
| paired seeds `t` | 5 |
| ceiling `max_attainable(5)` | `3.80169140625` |
| `THRESHOLD` = `1/alpha`, `alpha = 0.05/2` | `40.0` |
| `MIN_T_MIXTURE = 13` | seeds needed before ANY crossing is possible |
| decidable at this `t` | **no, by arithmetic fixed before the run** |

The word `undecided` here is a property of the schedule, not of the data: at five seeds the mixture cannot reach the threshold even if every seed lands at the clip bound `B = 2.0`.

## Consequence fidelity (1.7c): NOT MEASURED

Owner: Foreman (LOOP_PROMPT.md 1.7c). For the cells in the table above this is still true:
`scale/m3_quintuple.py` journals metrics and, for the ntr8192 `negation_scope` units, saved
no per-cell weights, so no trained settled/twin/argmax/glance weights exist to intervene on
AT THIS TABLE'S GEOMETRY. What exists on disk now: the e3-ladder units under
`results/m3_quintuple_v2_weights/` (60 checkpoints, of which the five shipped twin t\*=1 files
above are a verified subset) and results/phaseD_weights_*.pt, whose metrics.kind is
pivot_unsigned -- a different arm family, from scale/trained_projections.py.

## Weights shipped with this package

`weights/` carries five safetensors files, one per seed, all `twin` at `taske3_t1` (the
chain-family equilibrium task at t* = 1, geometry `k8_s64_d24_st150_ntr2048_nev2048_b21`):

| seed | file (`weights/`) | eval NRMSE (journal) |
|---|---|---|
| 0 | `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd0_taske3_t1.safetensors` | 0.9231181827 |
| 1 | `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd1_taske3_t1.safetensors` | 1.0785056996 |
| 2 | `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd2_taske3_t1.safetensors` | 0.9293004878 |
| 3 | `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd3_taske3_t1.safetensors` | 0.9568415797 |
| 4 | `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd4_taske3_t1.safetensors` | 0.9041007794 |

Seed mean **0.958373** (`results/e_ladder_reading.txt`). Seed 1 sits ABOVE predict-the-mean;
the t\* = 1 rung is underpowered (settled−twin −0.036025, CI [−0.118936, +0.062209], N=5),
so these tensors carry no capability claim -- they are the ship candidate's arm at the one
rung where both cells cleared the bar, exported for inspection and intervention work.

Provenance and verification, per file (`scripts/export_hf_weights.py`, manifest in
`weights/MANIFEST.json`): metrics matched by key against `results/m3_quintuple_v2.jsonl`
(journal commit `1cc7900`; export HEAD `1c56985`, both stamped into each file's metadata);
each tensor set reloaded from disk into a fresh `scale.m3_quintuple.QuintArm` and re-evaluated
on the task's own eval batch -- tensor round-trip drift exactly 0.0, |delta NRMSE vs journal|
exactly 0.000e+00 on all five shipped files, against an acceptance bar of 1e-6. A checkpoint
that fails is deleted and left out of the manifest rather than shipped with a caveat.
Compute context: the registered geometry pins `torch.set_num_threads(2)`
(`scale/m3_quintuple.py:74`) and every number here is a CPU-lane number; the CUDA lane is
opened but has produced no journalled figure yet.

## Limits

What this table does not cover, collected here rather than scattered through the rows. (a) EVERY TASK IN THE CORPUS IS STATIC. Both registered oracles are closed-form functions of the input with no fixed point: `negation_scope.oracle` is `x[:, p, CH_PAYLOAD] * x[:, f, CH_FLIP]` (scale/negation_scope.py:68) and `counter_squared_oracle` is `x[:, :, CH_FLIP].sum(dim=1) ** 2` (scale/negation_scope.py:127). A settling arm has nothing to settle toward on either, so the headline contrast was near-zero by construction and the NO DIFFERENCE verdict is evidence about these tasks, not about settling in general. (b) `counter_squared` HAS ZERO QUINTUPLE ROWS. It is registered in M3_TASKS and runs under `scale/m3_capability.py --task counter_squared`, but `scale/m3_quintuple.py` has no `--task` flag, so every number here is `negation_scope` at one geometry. (c) FIVE SEEDS CANNOT DECIDE ANYTHING ANYTIME-VALIDLY. `eprocess.MIN_T_MIXTURE = 13`; `max_attainable(5) = 3.80169140625` against `THRESHOLD = 40.0`. The intervals here are fixed-sample, read once, and carry no anytime validity. A real gap below roughly 0.05 NRMSE reads NO DIFFERENCE at five seeds whether or not it is real (M3_QUINTUPLE_PREREGISTERED_READING.md:87). (d) CONSEQUENCE FIDELITY (1.7c) IS AN EMPTY COLUMN. It needs trained weights per arm and the quintuple journals metrics only. (e) THE INTERVALS ARE MONTE-CARLO, AND THE PROSE DELIVERABLES CARRY A SECOND FAMILY. At five seeds the paired bootstrap distribution is finite (5**5 = 3125 resamples, 126 distinct values), so the exact percentile is computable and differs from the B=10000 draw: exact `[-0.042903167939657406, +0.03155692900564091]` against Monte-Carlo `[-0.04858658448547344, +0.03155692900564091]` for `settled - twin`. CHECKLIST.md:1167 and STATE.md:19 print the exact pair under the label `B=10000`. One further endpoint, `+0.146551` for `settled - softmax` at CHECKLIST.md:1168 and DONE.md:338, is in NEITHER family: it is absent from all 3125 exact values and from the 97.5th percentile at every Monte-Carlo seed 0..399. It is carried here as an open defect, not adopted. (f) COST IS NOT PRICED. `meta.seconds` in the journal is a contended box and is labelled PROVISIONAL by its producer; no wall-clock number enters this table.
