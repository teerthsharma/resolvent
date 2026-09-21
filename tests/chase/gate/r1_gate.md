# R1 -- clamp(u,0,1) has zero gradient outside (0,1): the two candidates, raced, and the kill

Every number below comes from `r1_gate.py` in this directory (`python r1_gate.py`,
scratchpad-only, `ceq/` untouched -- both new forms are a runtime monkeypatch of
`ceq.arm_smprime.magnitude`, the same technique `gate_init_repair.py` already
uses on `ceq.hf.train.build`). Raw per-stage output is `r1_results.jsonl`;
`r1_verdict.json` carries the final must-fire table and the kill verdict.

## (1) F1 reproduced first, before any model was touched

`default_rng(0)`, `d=512`, `n=20,000`, LayerNorm-normalised rows (biased
variance, no affine, no eps), `w ~ N(0, 1/d)`, float64. `u = layernorm(x) @ w +
bias`; since LayerNorm forces `sum x_i^2 = d` exactly, `u | x ~ N(bias, 1)`
regardless of `x`, which is the whole mechanism behind the two target cells.

| bias | P(u<=0) | P(u>=1) | frac carrying gradient | target |
|---|---|---|---|---|
| 0     | 0.5014 | 0.1602 | 33.84% | 0.501 / 0.160 / 33.8% |
| 0.999 | 0.1586 (not pre-registered; matches Phi(-0.999)=0.1587) | 0.4982 | 34.33% | -- / 0.498 / 34.3% |

Hard-concrete instance (zeta=1.1, gamma=-0.1 stretch) at bias 0: exact-0 =
0.855%, exact-1 = 0.860% (target 0.9% / 0.9%).

**All seven checks pass, within 0.001-0.004 of every quoted target.** The row
proceeds.

## (2) The two candidates, both prior art, both drop-in for `magnitude`

**Straight-through** (Bengio, Leonard, Bengio 2013): forward is `clamp(u,0,1)`
bitwise, backward is the identity. Verified bitwise against `torch.clamp` on
4,096 float64 values: `bitwise_equal = True`.

**Hard-concrete** (Louizos, Welling, Kingma 2018, L0 regularisation):
`clamp(sigmoid(u)*1.2 - 0.1, 0, 1)`, ordinary autograd through the sigmoid and
the clamp -- no custom backward, no stochastic gate term (this instantiates
the deterministic transform the F1 instance above measures).

Both installed as a monkeypatch of the module-global `ceq.arm_smprime.magnitude`;
`blend()` looks that name up at call time, so the whole call chain (`blend` ->
`hop` -> `numerator`/`operator`/`readout` -> `CEQAttention._smprime`) picks up
the replacement with no source file touched, and both are restored to the
original `clamp` immediately after each variant's run.

## (3) R7: the spurious-zero instrument, self-tested and cross-checked before it was trusted

Ground truth for "window (j,i] contains a true zero" is flag+cumsum
(Blelloch 1989 segmented scan): `flag_k = (m_k==0)`, `cum = cumsum(flag)`,
window has a zero iff `cum[i] > cum[j]`. Compared against the float
`path_product`'s own zero reads; a float-zero with no true zero underneath is
spurious (underflow, not a closed gate).

Self-test (S=16, one true zero at position 5, `m=0.7` elsewhere, float64 --
`0.7^16` is nowhere near underflow so every float zero here must be a true
one): `n_true_zero=55` (expected 55, exact match), `n_spurious=0`.

Cross-check against the task's own quoted defect count (constant `m=0.5`, no
true zero anywhere, isolating the underflow the instrument is built to catch):
S=64 float64 -> 0 spurious (0.5^64 does not underflow float64); S=4096 float64
-> **4,564,731 spurious**, an exact match to the quoted count. The instrument
reproduces the planted defect exactly and finds nothing where nothing is
planted -- it is not a check that cannot fail (see the closing section).

## (4) Trained at the reference config

`operator="smprime"`, float32, `h512/L4/H8/d_head64/seq512/batch8`, seed 0,
`data/tinystories_20k.txt`, 800 steps, `save_every=100`, `lr=3e-4`, AdamW
default betas/eps, device=cuda. Init is `gate_init_repair.py`'s own repair
(`m_head.bias=1-1e-3`, `theta_head.bias=1e-3`) via `build_repaired`,
byte-identical to that script's `repaired_build` -- the ONLY variable this row
changes relative to the repaired-clamp reference is `magnitude` itself.
`T.train()`/`T.build()` are reused (`ceq/hf/train.py`), not reimplemented.

`train_seconds` below is wall clock under heavy, documented multi-agent GPU
contention (this same session's Foreman and Cameron rows both name `r1_gate.py`
as one of 2-3 concurrent processes sharing an 8,188 MiB card at 95-100%
util for most of the session, and separately report 25+ minute starvation on
their own smaller jobs) -- it is NOT comparable to the reference's 206.9 s and
is reported for the record only, not as a cost claim.

Reference to beat (repaired clamp, quoted verbatim from the task and
independently re-derived here from `gate_init_repair_summary.json`'s own
`loss_curve` -- **1.3121604919433594 / 1.2664644169807433 / 0.03985696 /
0.31207275390625**, matching the quoted 1.3122 / 1.2665 / 0.0399 / 0.3121 to
rounding): loss_last **1.3122**, last-50 mean **1.2665 +/- 0.0399**, trained
exact-zero **0.3121**, trained max BACKWARD run **42**.

| | straight-through | hard-concrete | repaired-clamp (reference) |
|---|---|---|---|
| command | `python r1_gate.py` | `python r1_gate.py` | `gate_init_repair.py` |
| seed / dtype / device | 0 / fp32 / cuda | 0 / fp32 / cuda | 0 / fp32 / cuda |
| shape | h512/L4/H8/dh64/s512/b8 | h512/L4/H8/dh64/s512/b8 | same |
| steps | 800 | 800 | 800 |
| n_params | 13,130,772 | 13,130,772 | (same arch) |
| loss first -> last | 5.6708 -> **1.3669** | 5.6747 -> **1.1894** | 5.6557 -> 1.3122 |
| last-50 mean +/- std | 1.2999 +/- 0.0439 | **1.1391 +/- 0.0401** | 1.2665 +/- 0.0399 |
| frac grad nonzero, init | 1.0000 | 0.9978 | -- |
| frac grad nonzero, trained | 1.0000 | 0.9863 | -- |
| m_head.bias init -> trained | 0.9990 -> 0.9798 | 0.9990 -> 0.9714 | 0.999 -> 0.9797/0.9987/0.9908/0.9990 (4 layers) |
| **bias move (layer 0)** | **0.01923** | **0.02757** | ~0.001-0.021, either direction |
| exact-zero, init | 0.01825 | 0.00000 | 0.0182 |
| exact-zero, trained (no threshold, reported) | **0.40503** | **0.00061** | 0.3121 |
| spurious zeros, init / trained | 0 / 0 | 0 / 0 | -- |
| backward reach, init (mean/med/p95/max) | 79.07 / 52 / 244 / 498 | 256.50 / 256.5 / 487 / 512 | 0.672/0/3/10 |
| backward reach, trained (mean/med/p95/max) | 1.209 / 1 / 4 / **13** | 241.42 / 234 / 484 / **512** | **2.146/1/7/42** |
| train_seconds (contended, not comparable) | 849.8 | 1189.8 | 206.9 |

Reach is BACKWARD reach throughout: `G_ij = prod_{k=j+1}^i m_k`, so row `i`'s
live reach is the run of consecutive nonzero `m` ENDING at `i`, not the
maximal run through it either direction (that stretch metric overstates
reach ~1.7x, per the repaired-clamp row already measured).

**Straight-through collapses to almost the same bimodal, near-dead structure
as clamp itself** (40.5% exact zero trained, max backward reach 13, barely
above clamp's own 31.2%/42) -- an unbounded identity gradient does not stop
the optimiser from driving `u` deep outside `[0,1]`, where the clamp still
truncates the forward value to exactly `0` or `1`, and the model apparently
finds that useful. **Hard-concrete essentially never produces exact zeros**
(0.06% trained, the same order as its 0.86% at init) and the gate stays wide
open (median backward reach 234, near the full 512-token sequence) -- and
also reaches the lowest loss of the three by a wide margin.

## Pairing

Same seed, same data order (`T.ByteBatches` over the same corpus, same
`torch.manual_seed(0)` before `build_repaired`), same shape, same steps for
both variants. Verified with detrended (first-difference) per-step loss
correlation against a shuffled control:

real = **0.9621**, shuffled control = **-0.0298** (reference pair read
0.9832 / -0.0156 -- same order of magnitude and same sign structure, both
runs consumed the same batch sequence).

## The must-fire, checked exactly as pre-registered

| | straight-through | hard-concrete |
|---|---|---|
| (i) frac grad nonzero >= 0.95, init | PASS (1.0000) | PASS (0.9978) |
| (i) frac grad nonzero >= 0.95, trained | PASS (1.0000) | PASS (0.9863) |
| (ii) bias moves >= 0.05 | **FAIL (0.01923)** | **FAIL (0.02757)** |
| (iii) exact-zero at init < 0.05 | PASS (0.01825) | PASS (0.00000) |
| (iv) trained exact-zero, reported, no threshold | 0.40503 | 0.00061 |
| **all three (i-iii) pass** | **NO** | **NO** |

## THE KILL FIRES

Both parameterisations pass (i) and (iii) -- the gradient is alive at >=95%
of gate positions at both init and after 800 steps, for both forms, and
neither starts more than 1.8% exactly closed. **Both fail (ii)**: with the
clamp's zero-gradient mechanism removed entirely (straight-through) or made
almost irrelevant (hard-concrete's stretch pushes exact endpoints to <1%),
`m_head.bias` still moves only 0.019-0.028 over 800 steps -- inside the same
~0.02 band the ORIGINAL, frozen-gradient clamp run moved in
(`gate_init_repair.py`'s repaired-clamp bias went `0.999 -> 0.9797/0.9987/
0.9908/0.9990`, a spread of 0.001-0.021).

Per the pre-registration: **if both parameterisations fail the must-fire, the
closed-magnitude gate cannot be trained with live gradients at exact
endpoints, and exactness becomes EVAL-ONLY** -- train with hard-concrete,
freeze to clamp at eval. That is what this run found. Reported flatly, as
instructed.

**What this does NOT mean.** The kill is about `m_head.bias`'s own movement,
not about whether the gate's realised behaviour changes: exact-zero fraction
and backward reach both move by an order of magnitude or more between the two
forms (0.4x vs 0.0006x trained exact-zero; reach median 1 vs 234) and hard-
concrete reaches a materially lower loss (1.1391 vs 1.2665 vs 1.2999, last-50
mean). The bias scalar being pinned near its init is consistent with two
different explanations this run cannot separate: (a) the loss landscape has a
shallow, wide basin near `m~1` that AdamW at `lr=3e-4`/800 steps does not
escape regardless of gradient shape, or (b) the per-position `m_head` WEIGHTS
(not the shared bias) are doing the real work of opening/closing gates, with
the bias staying put because it does not need to move for the weights to
differentiate positions. Distinguishing those needs a run that tracks
`m_head.weight` norm growth, which R1 did not measure.

## What this does not establish

800 steps is 1.8% of one epoch over the corpus (same corpus, same caveat as
`gate_init_repair.py`'s own README): every number here is an early-training
snapshot, not a converged one. L-VARY (5 seeds, paired) was not run for this
row -- one seed each, matched to the existing repaired-clamp reference by the
pairing check above (itself a single-seed quote in the task). A 5-seed paired
table is the necessary next row before either form -- or the eval-only-clamp
fallback -- is adopted past a diagnostic.

`train_seconds` for both variants is inflated by documented, severe
multi-agent GPU contention (this session's own board log has other agents
naming `r1_gate.py` as a co-tenant and reporting 25+ minute starvation on
much smaller jobs); it says nothing about the two forms' relative compute
cost and is not compared against the reference's 206.9 s.

## Checks that cannot fail, named

`gate_init_repair.py`'s own `verify_repair` uses `frac_zero_exact < 0.05`,
matched to float32 on-disk precision (not a bf16 tolerance mismatch: this
run's own gate reads are float32, cast from the model's own dtype, the same
precision the threshold was set against). The spurious-zero self-test above
is built so a broken flag+cumsum (e.g. an off-by-one window) fails it outright
(`expected_true_zero=55`, not `0` or "all pairs" -- a LayerNorm-null-space
style degenerate case that a wrong implementation could pass by accident is
explicitly not what this self-test is), and it is separately cross-checked
against the task's own quoted S=4096 float64 spurious count (4,564,731, exact
match) rather than trusted on a synthetic case alone. The pairing control
(shuffled) is the one instrument whose job is to read near-zero; it is
reported beside the real correlation rather than alone, so a "passing"
control cannot be mistaken for a positive result. No zero-score-passes-the-bar
check exists in this row: must-fire (ii)'s threshold (0.05) is a fixed
constant compared against a measured float, not a formula that can be
satisfied vacuously, and it is the criterion that actually fired (both forms
measured well below it, 0.019 and 0.028) -- the row's kill rests on a check
that DID fail, not one that structurally cannot.
