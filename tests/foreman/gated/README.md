# The first trained gate, and why it does not measure the gate

Every one of the 165 checkpoints under `results/` is the ungated softmax corner:
the union of weight-key names across all of them is eight names, and 158 of 165
are `kind='softmax'` at `beta=0.5`. The gate this project is about had never been
trained and saved. This directory produces the first one.

The weights themselves are **not** in the tree — the checkpoint is 151 MB, and
`gated_run.py` rebuilds it in 207 seconds.

```bash
python gated_run.py
```

`operator="smprime"`, float32, `h512 / L4 / H8 / d_head 64 / seq 512 / batch 8`,
seed 0, over `data/tinystories_20k.txt`, 800 steps with `save_every=100`. Loss
`5.6557 → 1.5618`; 15,840.7 tok/s measured, 88.5% of the unshared rate because
another lane held the GPU. Float32 is not a choice: bfloat16 is rejected by
`torch.polar` and float16 crashes at step 0, so this is the only dtype the gated
arm runs in.

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

## The artefact

`model.safetensors` carries 73 keys, and all seven gate names are present on all
four layers: `m_head.{weight,bias}`, `theta_head.{weight,bias}`, `beta`, `qk`,
`g`. The three switches trained off their starting values, which no stored run
has ever recorded:

| layer | beta | qk | g |
|---|---|---|---|
| 0 | 1.049481 | 1.049030 | 0.978531 |
| 1 | 0.971136 | 1.067640 | 0.933655 |
| 2 | 0.928217 | 1.115753 | 0.919786 |
| 3 | 0.927931 | 1.063646 | 0.973957 |

## The gate is a hard truncation, not a decay

Read by monkeypatching `ceq.arm_smprime.blend` so the captured value is the `m`
the operator actually consumes — not `m_head.bias`, and not a re-derivation — on
a held-out `[8, 512]` byte batch, 16,384 gate values.

| | init | trained |
|---|---|---|
| exactly `0.0` | **49.3%** | **59.3%** |
| geomean over nonzero | 0.2425 | 0.9024 |
| p50 / p75 | 0.00807 / 0.31107 | 0.0 / 1.0 |

Two independent reads on different eval batches (seeds 999 and 12345) agree to
batch noise.

**The decay length is the wrong statistic for this operator.** `G_ij = ∏ m_k` is
exactly zero the moment a single `m_k` in the span is zero, so reach is set by
**runs of consecutive nonzero `m`**, not by a rate. Measured directly across all
16,384 positions:

| | mean run | median | p95 | max |
|---|---|---|---|---|
| init | 1.023 | 0 | 4 | 17 |
| trained | **0.685** | 0 | 3 | **7** |

**No row anywhere in the trained model carries a live path beyond 7 tokens**, at
a sequence length of 512. Training made the distribution *more* bimodal — more
exact zeros, and the survivors pushed harder toward 1 — rather than shifting a
graded decay.

Reported for completeness because they were pre-registered: `L = 1/ln(1/m̄)` on
the nonzero-conditioned `m̄` gives `L_init = 0.706` and `L_trained = 9.74`, so
both the `L < S/2 = 256` and `L < S/8 = 64` thresholds fire. Neither means
anything here, because `L_init` fires too. Using `m̄` over all entries gives
`L = 0` at both points, since one exact zero sends the geometric mean to zero.

## STRUCK: every number above measures a half-closed random start

`gate_init_repair.py` reruns this identically with `m_head.bias = 0.999` and
`theta_head.bias = 0.001` — the values `trainable_heads()` was written to set —
and changes nothing else. Both pre-registered verdict-change thresholds clear.

| | reference (broken init) | repaired | threshold |
|---|---|---|---|
| frac exactly 0.0 at init | 0.4915 | **0.0182** | < 0.05 |
| frac exactly 0.0 trained | 0.5969 | **0.3121** | < 0.45 |
| trained max run | 10 | **42** | > 14 |
| final loss | 1.5618 | **1.3122** | — |

The pairing is verified rather than assumed: `config.json` byte-identical, 73-key
state dicts identical in name set, `run_record` scalars identical, and detrended
per-step loss correlation **0.9832** against a shuffled control of **−0.0156**,
so both runs consumed the same batch sequence. Only the two biases differ.

**The open gate reaches a lower loss** — last-50 mean `1.2665 ± 0.0399` against
`1.5018 ± 0.0352` — so it is better at the task, not merely different.

**Data or init? Both, with init dominant.** Training still closes the gate hard
from a 98.2%-open start (`0.0182 → 0.3121`, stretch median `144 → 3`), so the
collapse is real. But the trained `m_head.bias` **barely moves from wherever it
starts** — reference `0 → −0.0172 / −0.0021 / +0.0068 / −0.0101`, repaired
`0.999 → 0.9797 / 0.9987 / 0.9908 / 0.9990` — so at 800 steps the bias is
effectively frozen at its initialisation and whatever it is handed is what it
keeps.

### Two corrections to the numbers above

**The `max 7` is eval-batch-dependent.** It was measured at `eval_seed=999`;
both training runs used `12345`, where the same reference checkpoint reads
`mean 0.941 / median 0 / p95 4 / **max 10**`. The ceiling is a property of the
eval draw as well as the model.

**"Stretch" is the wrong metric.** It assigns each position the length of its
whole maximal nonzero run, but `G_ij = ∏_{k=j+1}^{i} m_k`, so row `i`'s live
reach is the **backward** run ending at `i` — at most the stretch, and about half
on average.

| backward reach | mean | median | p95 | max |
|---|---|---|---|---|
| reference trained | 0.672 | 0 | 3 | 10 |
| repaired trained | **2.146** | **1** | **7** | **42** |

Stretch overstates reach by roughly `1.7×`. **Every reach number should be quoted
as backward reach.** The conclusion survives the stricter metric.

## Why half the gate is closed before training starts

The premise that the gate initialises near the softmax corner at `m = 0.999` is
true of `ArmSMPrime.trainable_heads()` and **false of the path this checkpoint
went through**. `CEQAttention.__init__` (`ceq/hf/modeling_ceq.py:439`) builds
`m_head` and `theta_head` as plain `nn.Linear`, and `_init_weights` normalises
every Linear weight and zeros every Linear bias generically. Nothing on the HF
path calls `trainable_heads()`; it appears only in `arm_phase.py` docstrings and
an inline copy at `ceqjepa/headtohead.py:171`.

With `g = beta = qk = 1` and bias `0`, `blend(u, θ, g=1) = clamp(u, 0, 1)` over a
`u` centred near zero, so roughly half the mass lands on the closed lower
endpoint from random initialisation alone. That is measured — 49.3% exactly zero
before any gradient step — not inferred. The trained `m_head.bias` values are
`-0.017239, -0.002079, 0.006750, -0.010053`, nowhere near `0.999`.

## What this does not establish

800 steps is **1.8% of one epoch** over the 18,167,706-token corpus, so every
gate number here is an early-training snapshot rather than a converged one. There
is no held-out score of any kind: `grep -ic eval ceq/hf/train.py` returns 0, the
only `data.batch()` call site passes `"train"`, and the `self.val` split the
loader builds is never requested. The corpus is children's stories, chosen for
being on disk rather than for carrying long-range structure, and nothing in
`ceq/` regularises `m`. So the run settles that a gated checkpoint can exist and
what the gate does early on a short-dependency corpus, and it settles nothing
about cost, refusal or architecture.
