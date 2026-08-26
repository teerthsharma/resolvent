# M3 — the deciding measurement, pre-registered before the run

Round 6, Phase C, RULE 2. Written before any cell of this run has produced a
number. Everything below is fixed from this point; the outcome table in section
5 is what each result licenses, and it is written now precisely so that it
cannot be chosen after the numbers are in.

Owner: Chase. Instrument: `scale/m3_quintuple.py`, built on the shipped
`scale/m3_capability.py` arm shape and the `scale/negation_scope.py` task.

---

## 1. The five cells

Round 5's plan was a triple. Foreman's ARM S report closed with a finding that
makes a triple unreadable:

    alpha near ONE-HOT even after the log-domain fix
    log_alpha min -182.7498   max -0.0

The fixed point is unique, is reached, and is lopsided. No birth gate asks
whether a lopsided equilibrium carries anything that a single highest-weight
pivot lookup does not. That question is T1's, pre-drafted, and it is cheaper to
answer it inside this run than to have a reviewer answer it afterwards. The run
is therefore a quintuple.

| # | cell | operator | settling |
|---|---|---|---|
| 1 | `softmax` | `bench._softmax_operator` | none — the baseline, reproduced first |
| 2 | `glance` | ARM S at `t_max = 0` | none — bitwise-bound to cell 1 by G3 |
| 3 | `settled` | ARM S | log-domain, `beta = 0.5`, `N = 21` |
| 4 | `twin` | ARM S architecture | none — `alpha` is the normalised gate |
| 5 | `argmax` | ARM S architecture | none — `alpha` one-hot at `argmax(gate)` |

Cells 3, 4 and 5 share every parameter tensor and differ only in how `alpha` is
obtained, so a contrast between them is not a contrast between architectures.

**Cell 2 is expected to be bitwise identical to cell 1.** `scale/arm_s.py:236`
returns the glance expression unchanged at `t_max = 0`, and that is the G3 bind.
Cell 2 is run anyway, at full cost, because a G3 bind taken at the money run is
worth more than an invented distinction between two cells that coincide. If cell
2 is not bitwise cell 1, the G3 bind has broken and the run is void.

**Cell 5 is the attribution control and it is not optional.** If `settled` beats
`twin` but ties `argmax`, the equilibrium bought nothing that the single
highest-weight pivot did not already carry, and the honest claim is routing-only.

---

## 2. Settings, fixed now

| setting | value | why this value |
|---|---|---|
| `n_train` | **8192** | at 2048 the softmax baseline fails its own absolute bar on 3 of 5 seeds (`0.949529 / 1.040708 / 1.045348 / 0.957720 / 1.042073`, mean `1.007076`), and the published `0.949529` is rank 1 of 5. The published 8192 figure is `0.877168`. |
| `n_eval` | 512 | as published |
| `s`, `d` | 64, 24 | the harness defaults the published figures were taken at |
| `steps`, `lr` | 150, 0.02 | as published |
| seeds | **0, 1, 2, 3, 4 — all five printed** | never a single figure |
| `k` | **8 for the money cell**; 16 and 32 as a follow-on | see section 7 |
| `beta` | 0.5 | `kappa(T) = beta` exactly and attained |
| `N` (Neumann) | 21 | `neumann_for(0.5)`, known before the run rather than fitted after it |
| threads | 2, pinned in the probe file | the same command at 2 and at 20 threads moved a published reading from `0.747528` to `0.747062` |

`k = 128` is excluded, pre-registered, on two independent grounds: it fails K-F
on FLOPs (`3.851464` against `1.667480`), and round 5 measured Karcher
uniqueness on only `0.5167` of draws there, which is outside the uniqueness-safe
regime the certificate is stated in.

---

## 3. The resolution floor — stated BEFORE, so that a null is interpretable

Iteration 0's synthetic dry-run measured what this harness can and cannot see at
these settings, using arms whose answer was known in advance:

* a planted gap of `+1.005203`, CI `[+0.967798, +1.040972]`, was detected — the
  harness reported `SETTLED WINS`, correctly;
* two null arms were not called a win — a bitwise-identical arm (`delta` exactly
  `0.000000`) and an arm carrying an information-free extra feature (`delta
  +0.005522`, CI `[-0.027900, +0.057033]`).

The per-seed paired delta standard deviation was `0.056889` for the genuinely
paired null and `0.057089` for the unpaired one. **Pairing bought almost
nothing**, because an arm with an extra parameter takes a different Adam
trajectory and the shared variance does not cancel.

**Therefore: with five seeds, a real settled-vs-twin gap below roughly 0.05
NRMSE will read `NO DIFFERENCE` whether or not it is real.** A null in this run
is evidence of "no effect larger than about 0.05 NRMSE"; it is not evidence of
"no effect". This sentence is written before the run and is repeated verbatim
beside any null result.

---

## 4. The statistic

For each contrast, the paired bootstrap over training seeds already built and
tested at iteration 0 (`scale/m3_synthetic_settled.py::contrast`, bound by
`tests/chase/test_m3_synthetic_settled.py`):

    delta = mean_over_seeds( NRMSE_reference(seed) - NRMSE_arm(seed) )

Positive `delta` means the arm has the lower error. `B = 10000` percentile
bootstrap over the paired per-seed differences. Verdict, strict at zero (G6):

    ci_lo > 0  ->  arm wins
    ci_hi < 0  ->  reference wins
    otherwise  ->  NO DIFFERENCE

The shipped marginal `negation_scope.bootstrap_ci` is not the instrument for any
contrast here. It resamples one eval batch from one training run, and the repo
already records that its width is narrower than the same arm's seed-to-seed
spread, which makes an "intervals overlap" verdict decidable by the choice of
seed. Marginal intervals are printed per cell as description only.

**Headline cell: `settled` vs `twin`.**
**Headline caveat: `settled` vs `argmax`.**

---

## 5. The outcome table — written now, not after

| result | what it licenses |
|---|---|
| `settled` beats `twin` (CI excludes 0) and beats `argmax` (CI excludes 0) | the equilibrium is doing work no single pivot lookup does. +12. |
| `settled` beats `twin` but ties `argmax` | the equilibrium bought nothing over the argmax lookup. Claim rewritten routing-only, +6. The word "equilibrium" leaves the claim sentence. |
| `settled` ties `twin` | K-C fires. The equilibrium clause is cut project-wide, +6 ceiling, T1 released. |
| `settled` loses to `twin` | the settling is a regression; reported as such, no positive claim. |
| any cell whose seed-mean NRMSE is at or above 1.0 | that cell did not beat predict-the-mean and is credited with nothing, whatever its contrast says. |

The last row binds cells 3, 4 and 5 as hard as the first four rows do. A
contrast between two arms that both fail the absolute bar is a comparison of two
failures.

---

## 6. What will NOT be claimed

**No wall-clock number is evidence in this run.** K-F is UNDECIDED, not passed.
Foreman measured FLOP ratios `1.010420` to `3.851464` against clock ratios
`1.6546` to `45.0608` on a contended box, and fetched the accounting rather than
asserting it: arXiv 2302.06117 on framework-boundedness, pytorch#41383 on
per-operation overhead, and NOT FOUND any document claiming `torch.compile`
removes per-op dispatch overhead on CPU. Closing K-F needs `collect_callgrind`
or an isolated core, and this box has neither. Cost is reported as analytic
FLOPs (`scale/m3_flops.py`) or not at all. Any seconds figure appearing in a log
is labelled provisional and is not carried into a verdict sentence.

Every projection in ARM S is random-init at the time of writing. This run trains
them, so it is the first M3 reading of ARM S that is not random-init. The
separate open item — that every *probe* number in the project was taken at
random-init — is untouched by this run and remains open.

---

## 7. Feasibility, stated before the deadline rather than at it

The settling map in `scale/arm_s.py` is written per example. M3 trains on
`[n, s, d]` with `n = 8192`, so a Python loop over examples inside the training
loop is `8192 x 150` settle calls per unit and is not affordable. The map is
batched in `scale/arm_s_batched.py` and bound bitwise against the per-example
original before any cell runs; if that bind fails, the run does not start.

The full grid — 5 cells by 5 seeds by 3 values of `k` — does not fit in the
iterations RULE 2 allows. The run is therefore ordered:

1. **k = 8, all five cells, all five seeds — the money run.** 25 units.
2. `k = 16` and `k = 32` as a follow-on bucket, which is iteration 13's ablation
   ladder rather than the deciding measurement.

Bucketed through `scale/bucket.py` (ADR-001); one unit is one (cell, k, seed).
Every unit is journalled to `results/m3_quintuple.jsonl`, and a partial set is
reported as partial, with its bucket count, rather than aggregated.
