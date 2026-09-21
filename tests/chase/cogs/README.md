# The COGS win bar is void

`ceq/capability.py:39-40` sets `WIN_MARGIN = 0.02`, and `results/capability.json`
records a softmax control at COGS generalization `0.02930`. Together those give
a pre-registered win bar of `0.04930` for any challenger architecture. COGS is
the only capability split in this repository whose floor holds — in-distribution
softmax `0.92578` against `RESOLUTION_FLOOR = 0.20` — so it was the one bed where
a null would have meant something.

The recorded control ran **3,000 steps**, which at batch 32 and sequence 192 is
18,432,000 tokens, about **3.97 epochs** of the 24,155-item COGS train split.
COGS generalization is exactly the quantity that keeps improving long after
in-distribution saturates. This directory trains the identical control longer.

```bash
python cogs_control.py
```

Run from the repository root so `ceq.harness` and `ceq.lm` import. One continuous
12,000-step run, `softmax`, seed 0, `bs=32`, `lr=3e-4`, `seq=192`, `d=256`,
`n_heads=4`, `d_head=64`, `n_layers=4`, `vocab=874`, `n_params` confirmed
`3,652,096` by `numel()`, evaluated at four checkpoints with
`gate_eval=256, gen_eval=512, max_new=192` — the same protocol as
`ceq.capability.BUDGET["cogs"]`.

## The curve

| step | in-distribution | generalization | train loss |
|---|---|---|---|
| 3,000 | 0.93359 | **0.02148** | 0.2731 |
| 6,000 | 0.94531 | **0.13672** | 0.2887 |
| 9,000 | 0.94531 | **0.14648** | 0.2464 |
| 12,000 | 0.94922 | **0.09766** | 0.2331 |

**The same control sits 2.0×–3.0× above the 0.04930 bar at every checkpoint from
6,000 steps on.** A challenger that beat `0.04930` would lose to this control
given more budget, so the recorded pairing measures training length rather than
architecture.

In-distribution was never the undertrained quantity: it is saturated from step
3,000 against a `score_ceiling` of `0.951171875`.

The curve is a property of the weights, not of logging. All four saved
checkpoints were re-scored in a separate process and returned the same counts —
`11/512, 70/512, 75/512, 50/512` generalization and `239/242/242/243` of 256
in-distribution. Paired worst-case McNemar puts the 3k→6k rise at `χ² ≥ 43.0`,
`p < 1e-10`, and the 9k→12k fall at `χ² ≥ 5.0`, `p ≤ 0.025` even if the solve
sets were disjoint.

## The bar cannot be replaced yet

The curve has **not flattened** by 12,000 steps and it is **non-monotone** — it
peaks at 9,000 and falls at 12,000 — so every candidate value is a point the run
has already passed through in both directions. Setting a new bar needs either a
longer run or a learning-rate schedule, and the honest statement today is that
the old bar is void and no replacement exists.

## Two defects found on the way, both in the checking rather than the result

**The reproduction check is only half able to fail.** The tolerance was
`|measured − recorded| ≤ 0.03` per split, stated as a constant before the run,
giving bands `[0.896, 0.956]` and `[0.000, 0.059]`. The in-distribution half
fired live: attempt 1 returned `0.81250`, delta `−0.11328`, and the script halted
before touching the curve; attempt 2 of the identical command returned `0.93359`.
But the generalization half **cannot fire low** — `|0 − 0.029297| = 0.0293 <
0.03` — so a model scoring exactly `0.0` on generalization would "reproduce" the
control. That is the same disease a prior lane identified in this bed's own
checks, surviving in half of its replacement.

**Training is not reproducible run to run.** Three of four seed-0 CUDA runs of
the identical recipe land inside a `±0.03` band and one does not, a swing of
`0.113` absolute at a fixed seed. SDPA kernel selection and embedding-gradient
scatter-add are both nondeterministic here. Any single-run comparison on this bed
inherits that noise floor.

**And `ceq/harness.py:313` ignores its own seed argument.** `eval_indices` seeds
off `n_test` alone, so eval seeds 1 and 2 return byte-identical scores.
Checkpoint comparisons are therefore perfectly paired, which is good, but no
subsample-luck estimate is reachable through that knob, which is not.

> **PROVISIONAL — PHASE I.1 R1.** Every gate number below was measured under
> `magnitude = clamp(u, 0, 1)`, whose gradient is **zero outside (0, 1)**. At
> bias 0, `P(u ≤ 0) = 0.501` sit at an exact zero with no gradient; at bias
> 0.999, `P(u ≥ 1) = 0.498` sit at an exact one with no gradient. Only **33.8%**
> and **34.3%** of gates carry gradient at the two starts respectively, so the
> bias gradient — a mean over per-gate gradients — is small either way. "Frozen
> at initialisation" is the **clamp**, not the data, and a gate that reaches
> exactly zero under clamp **never reopens**: dead, not closed. These rows are
> re-measured under R1 (straight-through or hard-concrete) before any sentence
> about the gate stands.

## What was not run

The gated arm was not raced here — racing against an unsettled bar would settle
nothing. Measured separately this session, the `smprime` arm reads generalization
`0.0` and in-distribution `0.16797`, below `RESOLUTION_FLOOR = 0.20`, so it fails
the bed's own admission gate and dies under any bar, valid or not.

The whole curve rests on one training trajectory at one seed, on a bed whose
seed-0 in-distribution swung `0.12` absolute between runs.
