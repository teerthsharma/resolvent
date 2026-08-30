# Capability table v0 -- CEQ signed pivot-routed attention

Task `negation_scope` at `s64_d24_st150_ntr8192_nev512_b21`, 5 seeds, `n_params = 4769` on every arm. Journal `m3_quintuple_v2.jsonl` at commit `551d512` (HEAD `8dd9a99`). Pre-registered reading: `M3_QUINTUPLE_PREREGISTERED_READING.md`. Produced by `scale/m3_quintuple.py --cells softmax glance settled twin argmax --seeds 0 1 2 3 4 --ks 8`.

**Softmax is the baseline and is measured first.** A verdict of NO DIFFERENCE is printed wherever that is what the interval says.

**This package ships no language-model weights, and its table cells carry no weights either.** The numbers below come from `results/m3_quintuple_v2.jsonl`, which journals metrics and not tensors. What IS shipped is a separate, verified set of trained probe-arm tensors under `weights/` -- 5 4,769-parameter `twin` checkpoints at task `e3_t1`, one per seed. See "Weights shipped with this package" below; none of them loads into `CEQForCausalLM` and none is named `model.safetensors`.

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

**Every row carries BOTH interval families, each under its own name, and the two differ.** At five seeds the paired resample space is finite, so the percentile the Monte-Carlo draw estimates is also computable outright: the `exact 95% CI` column is the exact percentile over all 5**5 = 3125 paired resamples. It is not a second measurement and not a correction -- it is the same per-seed deltas under a different resampling rule, and a headline quoted from one family will not match the other. Prose deliverables in this repository adopted the exact family (`CHECKLIST.md:1239`); this card prints the Monte-Carlo family the run executed and names both, so a reader who finds two endpoints for one headline can tell which instrument produced each instead of assuming one is a typo.

Both families are computed from the same 10 journal records in `m3_quintuple_v2.jsonl`, seeds 0, 1, 2, 3, 4 -- for the row below, keys `<arm>_k<k>_s64_d24_st150_ntr8192_nev512_b21` and `<reference>_k<k>_s64_d24_st150_ntr8192_nev512_b21` with the `_sd<seed>` suffix over those seeds. No number in this section is transcribed from another document.

| arm | reference | delta | 95% CI (Monte-Carlo) | exact 95% CI | seeds favouring arm | verdict | note |
|---|---|---|---|---|---|---|---|
| `settled` | `twin` | -0.002959 | [-0.048587, +0.031557] | [-0.042903, +0.031557] | 3/5 | **NO DIFFERENCE** | HEADLINE |
| `settled` | `argmax` | +0.226893 | [+0.175040, +0.275862] | [+0.175040, +0.276921] | 5/5 | **settled WINS** | HEADLINE CAVEAT -- argmax is the attribution control |
| `settled` | `softmax` | +0.108437 | [+0.066232, +0.147110] | [+0.068181, +0.147110] | 5/5 | **settled WINS** |  |
| `twin` | `softmax` | +0.111396 | [+0.100873, +0.121920] | [+0.100873, +0.121920] | 5/5 | **twin WINS** |  |
| `argmax` | `softmax` | -0.118456 | [-0.134115, -0.102204] | [-0.134115, -0.102786] | 0/5 | **softmax WINS** |  |
| `glance` | `softmax` | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] | 0/5 | **NO DIFFERENCE** | G3 binds glance bitwise to softmax at t_max = 0: this row is zero BY CONSTRUCTION, not a measured tie |

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

Owner: Foreman (LOOP_PROMPT.md 1.7c). scale/m3_quintuple.py NOW SAVES per-cell weights (results/m3_quintuple_v2_weights/, one .pt per unit carrying the state_dict plus every constructor argument and the mu/sigma standardisation), so the blocker this field used to name is gone for units run from that change onward. IT IS NOT GONE FOR THIS TABLE. The 25 negation_scope units this table is built from were journalled BEFORE the change and carry no tensors; 0 of the 25 weight files this table would need are on disk. Re-running them to emit weights costs 6685.3 s by their own meta.seconds and has not been paid. The only other trained weights on disk are results/phaseD_weights_*.pt, whose metrics.kind is pivot_unsigned -- a different arm family, from scale/trained_projections.py.

## Weights shipped with this package

`weights/` carries 5 safetensors files, one per seed, all `twin` at task `e3_t1`, geometry `k8_s64_d24_st150_ntr2048_nev2048_b21`:

| seed | file (`weights/`) | eval NRMSE (journal) |
|---|---|---|
| 0 | `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd0_taske3_t1.safetensors` | 0.9231181827 |
| 1 | `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd1_taske3_t1.safetensors` | 1.0785056996 |
| 2 | `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd2_taske3_t1.safetensors` | 0.9293004878 |
| 3 | `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd3_taske3_t1.safetensors` | 0.9568415797 |
| 4 | `twin_k8_s64_d24_st150_ntr2048_nev2048_b21_sd4_taske3_t1.safetensors` | 0.9041007794 |

Seed mean **0.958373**. These tensors carry NO capability claim: they are the ship candidate's arm at the one rung where the cells cleared predict-the-mean, exported for inspection and intervention work. Read the rung's own interval before quoting any of them.

Provenance, per file (`scripts/export_hf_weights.py`, manifest in `weights/MANIFEST.json`): metrics matched by journal key against `results/m3_quintuple_v2.jsonl`, then each tensor set reloaded from disk into a fresh `scale.m3_quintuple.QuintArm` and re-evaluated on the task's own eval batch. Worst |delta NRMSE vs journal| over the 5 shipped files is 0.000e+00, against an acceptance bar of 1e-06. A checkpoint that fails is deleted and left out of the manifest rather than shipped with a caveat, so an unverified entry never reaches this table.

## Limits

What this table does not cover, collected here rather than scattered through the rows. (a) EVERY TASK IN THE CORPUS IS STATIC. Both registered oracles are closed-form functions of the input with no fixed point: `negation_scope.oracle` is `x[:, p, CH_PAYLOAD] * x[:, f, CH_FLIP]` (scale/negation_scope.py:68) and `counter_squared_oracle` is `x[:, :, CH_FLIP].sum(dim=1) ** 2` (scale/negation_scope.py:127). A settling arm has nothing to settle toward on either, so the headline contrast was near-zero by construction and the NO DIFFERENCE verdict is evidence about these tasks, not about settling in general. (b) `counter_squared` HAS ZERO QUINTUPLE ROWS, and not for want of a flag. It is registered in M3_TASKS and runs under `scale/m3_capability.py --task counter_squared`; `scale/m3_quintuple.py` now registers `--task` too, and 61 of the 86 rows in `results/m3_quintuple_v2.jsonl` carry an e3 task suffix written through it (e3_t1 16, e3_t2 15, e3_t32 15, e3_t8 15), against 25 bare `negation_scope` keys. `counter_squared` was simply never run under the quintuple. Every number in THIS table is `negation_scope` at one geometry because `read_journal` filters on `m3_quintuple.task_of`, not because the corpus is all the producer can emit. (c) FIVE SEEDS CANNOT DECIDE ANYTHING ANYTIME-VALIDLY. `eprocess.MIN_T_MIXTURE = 13`; `max_attainable(5) = 3.80169140625` against `THRESHOLD = 40.0`. The intervals here are fixed-sample, read once, and carry no anytime validity. A real gap below roughly 0.05 NRMSE reads NO DIFFERENCE at five seeds whether or not it is real (M3_QUINTUPLE_PREREGISTERED_READING.md:87). (d) CONSEQUENCE FIDELITY (1.7c) IS AN EMPTY COLUMN. It needs trained weights per arm and the quintuple journals metrics only. (e) THE INTERVALS ARE MONTE-CARLO, AND THE PROSE DELIVERABLES CARRY A SECOND FAMILY. At five seeds the paired bootstrap distribution is finite (5**5 = 3125 resamples, 126 distinct values), so the exact percentile is computable and differs from the B=10000 draw: exact `[-0.042903167939657406, +0.03155692900564091]` against Monte-Carlo `[-0.04858658448547344, +0.03155692900564091]` for `settled - twin`. CHECKLIST.md:1167 and STATE.md:19 print the exact pair under the label `B=10000`. One further endpoint, `+0.146551` for `settled - softmax` at CHECKLIST.md:1168 and DONE.md:338, is in NEITHER family: it is absent from all 3125 exact values and from the 97.5th percentile at every Monte-Carlo seed 0..399. It is carried here as an open defect, not adopted. (f) COST IS NOT PRICED. `meta.seconds` in the journal is a contended box and is labelled PROVISIONAL by its producer; no wall-clock number enters this table.
