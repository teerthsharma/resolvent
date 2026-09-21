# Phase I — Calibrate and Cost

Phase I asked two questions: whether calibration can carry an architecture claim,
and what a training run of this model would cost. Both were answered, and both
answers point at the same absence.

The short form: **reliability is free and cannot carry a claim; the degree law
that would beat attention is exact and has never been raced against attention;
the exact-zero refusal certificate is underflow at every dtype; and no checkpoint
in this repository has ever been trained with the gate all three claims are
about.**

---

## 1. The calibration bed, and what it ruled before any arm ran

A 28-state absorbing chain with `cond(I − Q) = 4.216` and Monte-Carlo standard
error falling at slope `−0.5017` against the theoretical `−0.5`.

**Two corrections to how that bed was first reported, both found by re-checking
rather than by re-running.** Neither the bed nor its `2.776e-17` Murphy residual
exists anywhere in the tracked repository — a tree-wide search returns no match,
and both live only in an uncommitted scratchpad, which is the same defect that
Phase H §8 carried until its producer was landed. And the residual is not a
verification: it is a **bin-substitution tautology**, the decomposition checked by
substituting back the same bin probabilities it was built from, so it could not
have failed. A quantity that cannot fail is not evidence, and this project's own
standing rule — a bar whose branch never fires is worth nothing — applies to it.

What survives is the structural ruling, which does not depend on that residual.

The bed's own credibility check fired immediately. A **base-rate forecaster
scores reliability `0.0`, the best of every arm, and resolution `0.0`, the
worst**: perfectly calibrated and perfectly useless. So a bar on reliability
alone admits the one forecaster guaranteed to be worthless, and the bar was
written jointly instead:

```
reliability_CI_upper <= 0.001299   AND   resolution_CI_lower > 0.0
```

Only `exact_read` and `noisy_oracle_sigma0.05` pass it.

### Reliability is free, and this is a theorem rather than an observation

Temperature scaling is **monotone on logits**. Resolution scores the ranking,
which a monotone map preserves exactly. So temperature scaling *cannot move
resolution* and *can zero reliability*, by construction.

Measured on a deliberately mis-scaled arm: reliability `0.02407 → 0.00012`, a
factor of **200**, at a fitted `T = 2.742` against an injected `k = 2.5`, while
resolution stayed flat across `0.044–0.046`. Foster–Vohra says the same thing
from the other side: calibration is achievable online by a recalibrator with no
forecasting skill at all.

One sentence carried into this phase from an earlier draft was simply wrong and
is struck here. A proper scoring rule is optimised by the **true conditional
probability** (Gneiting and Raftery 2007), not by "the calibrated forecast".

**The consequence for any future architecture claim.** It must live in
resolution, with the recalibrator applied to *both* arms first so reliability is
equalised. And the gap must clear a real margin: recalibrator drift on this bed
is `2.0–2.2%`, while the sharp-versus-mis-scaled resolution difference is
`0.04545` against `0.04400`, which is `3.2%` — **inside the drift**. A resolution
gap under roughly 5% of resolution is noise.

### The conformal row is void on its own bar

Split conformal coverage is `≥ 1 − α − 1/(n_cal+1)` and is **marginal, not
conditional**. The pre-registered bar treated it as a per-seed floor; measured
across seed × α it holds `7 of 15`. That is not a failure of the method, it is
the bar asking the wrong question, and the ruling is recorded as:

> a marginal guarantee licenses a population-level refusal **rate**, never a
> per-instance refusal **claim**.

Group-conditional coverage ranges `0.8834–0.9297` against a nominal `0.90`. Both
degenerate arms pass trivially and are reported as such: base-rate Brier with
`REL 4.9e-32 / RES 0.0`, and an always-full-8-class predictor at coverage `1.0`,
set size `8.0`.

The NEVER leg does fire honestly: far state 12, `0/3000` reached, rule-of-three
upper bound `3/3000 = 0.00100`.

### There is a real decision stream, and it fails at its floor

The sharper question — whether this operator can be scored by a proper rule at
all — has an answer, and it is not the retirement it looked like. **Chess
self-play is a real decision stream**: `ceqjepa/beds/chess.py` and `chess_do.py`
produce exact four-way categorical outcomes with observational and forced-move
`do()` arms, already consumed by `ceqjepa/causal_eval.py`. Markets are the
opposite and the canon already says so: verse 09.20 records no market data, bed,
label, floor or baseline in the tree, re-verified here against `data/`.

Only one of the operator's three cited properties can carry a proper score.
`committor()` (`ceqjepa/operator.py:237`) returns a row-stochastic vector over
absorbing sets — it already *is* a forecast distribution. Exact refusal is a
raised `SingularTransientBlockError`, and a raised exception is not a point on
the simplex. Decidable NEVER is a proof-checker fact with no probability
anywhere in its statement. Neither can be repaired into a proper-score argument
as built; they gate *which* items receive a forecast, they are not forecasts.

**And at its shipped settings the chess bed cannot discriminate.** At
`max_plies=80`, **98.67% of games end in SINK**, and the *oracle* resolution
ceiling — the hard upper bound for any forecaster, operator or softmax or
perfect — is `0.000501`, CI95 `[6.09e-05, 1.04e-03]`, with `RES/UNC = 1.9%`.
Against that, a genuinely discriminating 8-bin arm posts `0.000645` while a
1000-permutation shuffle null posts `0.000579` — analytic `(B−1)·UNC/N` predicts
`0.000572`, agreeing to 1.2% — for a ratio of `1.11×` at **`p = 0.337`**. It does
not clear chance. That is the fifth bed in three days to fail at its floor rather
than at its model.

**It is a reprice, not a retirement, and the fix is one keyword.** `max_plies` is
already an argument of `generate_selfplay_game` and `ChessBed.build`. Same
generator, same seed, `80 → 400`: the ceiling moves to `0.101170`, CI95
`[0.0708, 0.1329]` — **202×, non-overlapping intervals** — `RES/UNC` rises to
`18.0%`, and the same arm clears the null at `3.99×`, `p = 0.000`. The cost is
**636 seconds**.

Nothing in this tree has ever computed reliability and resolution on the
operator's own `committor` output. The Murphy code has zero `ceqjepa` imports and
has only ever run on synthetic Beta(2,2) data. That is still true at the end of
this phase; what changed is that it is now known which bed to run it on, and at
what setting.

---

## 2. The exact-zero refusal is underflow, at every dtype

The project sold an exact zero in the gate path product `G_ij = ∏ m_k` as a
differentiator over a thresholded sigmoid: a threshold is a tunable constant, a
zero is not. Measured on the shipped `path_product` with a **constant gate and no
closed gate anywhere**, so every reported zero is spurious by construction:

| dtype | S | gate | entries reading bitwise `0.0` |
|---|---|---|---|
| bfloat16 | 64 | 0.2002 | **21** |
| float32 | 64 | 0.15 | **45** |
| float64 | 4096 | 0.5 | **4,564,731** |

At `m = 0.5` the first spurious zero appears at **S = 135** in bfloat16,
**S = 151** in float32, **S = 1076** in float64. Across a 16-cell grid — S in
`{64, 256, 1024, 4096}` crossed with four dtypes — the shipped product is wrong
in **12 of 16 cells**. Float64 is a later wall, not an absent one.

The failure mode is not a missed refusal. It is a **manufactured refusal**: the
certificate forbids a path that exists.

**One qualification, and it cuts both ways.** These counts are measured on
`path_product` in isolation. The full gated forward pass cannot reach bfloat16 at
all: `gate()` calls `torch.polar`, which rejects bfloat16, so `hop`, `numerator`,
`operator` and `ArmSMPrime.forward` all raise `RuntimeError: Expected both inputs
to be Half, Float or Double tensors but got BFloat16`. Only the modulus row alone
survives there. So the bfloat16 column describes a primitive rather than a shipped
forward, and the phrase "exact on shipping primitives" fails for a second,
independent reason — the shipping primitive is float32, at roughly 3.7× the cost,
because the complex dtype forces it. `float16` is accepted and silently returns
**complex128**, an eightfold precision inflation that nothing asked for.

The float32 and float64 columns are unaffected by any of that, and float64 at
S=4096 is where the certificate fails worst.

Compounding is not the mechanism and cannot be bounded away —
`prefix_logit_mask_restated` proves `|pathProd| ≤ 1`, so no gain above one exists
and a small leak never grows. The mechanism is plain underflow of a product of
numbers below one.

`zero_hop_mask` (`ceq/arm_smprime.py:788`) asserts that "a 1e-300 product does not
read as a zero it is not". In float32, `1e-300` is below the smallest subnormal
and is not representable at all.

### The replacement, bound and then priced honestly

Whether a gate closed is a Boolean fact, and a Boolean's prefix is an integer:
`b_k = (m_k == 0)`, `seg_k = cumsum(b_k)`, and `G_ij == 0 ⟺ seg_i ≠ seg_j`. A
closed gate flips `seg`; many small numbers multiplied never touch it. No
threshold, no prefix logarithm — `log(0) = −∞` poisons the whole prefix, which is
the standing `no_prefix_scan_represents_a_zero_gate` result.

Bound against the shipped code, RED on the float route and GREEN on the segment
route in all three:

| test | shipped `path_product` | `seg` | ground truth |
|---|---|---|---|
| S=64, bf16, no closed gate | 21 refusals | **0** | 0 |
| S=64, one gate at k=32 | — | exactly **1024** pairs | 1024 |
| S=4096, three gates | — | **bit-for-bit** | — |

The shipping leg was measured as segment-wise SDPA against dense
reachable-key-masked SDPA on the same Q/K/V: `max_abs_diff = 2.384e-07` against a
`5e-2` tolerance. `seg` reports zero spurious refusals in all 16 dtype cells.

**And it is prior art.** The construction is Blelloch's segmented scan (IEEE
Trans. Computers 38(11), 1989; CMU-CS-90-190, 1990) applied unchanged — the
combinator `⟨a,p⟩ + ⟨b,q⟩ = ⟨a·¬q + b, p ∨ q⟩` is algebraically the same test.
`flash_attn_varlen` already provides exact integer-indexed block-diagonal
reachability in shipping bf16 kernels. Gated linear attention already ships a fix
for this same underflow symptom, in log-space cumulative sums.

What survives subtraction is one scoping result, stated at its real size: a
**hard-clamped `max(0, ·)` gate is the one gate family that emits a true
algebraic zero**, so it is the one where the flag reads directly off `m_k == 0`
with no threshold. Mamba's `Δ → ∞` is asymptotic and gated linear attention's
gates are sigmoid; neither ever reaches zero, so neither has a bit to scan. The
boundaries in `cu_seqlens` are supplied as external document metadata —
*index-from-gate* rather than *index-from-metadata* was not found already
published.

The 1989 IEEE paper itself is recorded **UNREACHED** after two DNS failures; the
attribution rests on Blelloch's own tech report, read directly.

---

## 3. The degree law is exact, and it was never the obstacle

The Chebyshev degree triple `11 / 34 / 130` is indexed by **γ, the discount**, at
`γ = 0.50 / 0.90 / 0.99`. It has no dependence on sequence length. From the
Bernstein ellipse at `a = 1/γ`, with `ρ = 1/(a + √(a² − 1))`:

```
K = ceil( ln( 2C / ((1-rho) * eps) ) / ln(1/rho) ) - 1,    C = (1 + rho^2) / (1 - rho^2)
```

reproduces all three recorded degrees **exactly**, with no fitted constant, at
`ε = 1e-6`. So `K ≈ ln(1/ε)·√(1/(1−γ))` — Chebyshev's square-root acceleration —
and `K` is `O(1)` in sequence length.

Stated as a trade rather than as a speedup: **attention buys an unbounded horizon
at quadratic cost in length; the filter buys a tunable horizon at linear cost in
length, and the exchange rate is √horizon.** At bandwidth `b`, the filter passes
`N²` at `N > K·b` — `N = 34` for a 10-step horizon, `130` for 100, `463` for
1000.

The struck cost lever was filter against a **dense solve** at density 0.1, where
the solve won 6 of 6 cells by `2.75×` to `54×`. That is a different comparison.
**The filter has never been raced against attention.**

The condition that decides it is bandwidth, and bandwidth is the gate's decay
length `L = 1/ln(1/m̄)`. A learned gate drifting to `m̄ → 1` makes the operator
dense and inverts the result — which is where this phase ran out of evidence, for
the reason in §5.

---

## 4. Cost, measured rather than rescaled

The published hours table did not compute cost; it rescaled a prediction made at
an assumed flat 30 TFLOPS. Recomputed from `6·N·D` against measured achieved
throughput on an RTX 4060 Laptop (8188 MiB, cc 8.9, torch 2.14.0+cu126, batch 32,
seq 512, vocab 256, seed 0, 5 warmup + 20 timed):

| N | D | measured bf16 | fits a 9-hour cap |
|---|---|---|---|
| 12M | 240M (20×) | 0.354 h | yes |
| 25M | 500M (20×) | 1.535 h | yes |
| 50M | 1.0B (20×) | **4.915 h** | **yes, 45% margin** |

The 50M row does not flip. It exceeds the cap only in float32, at 18.560 h.

Two corrections came with it. The "67,108,864-token corpus" is the `max_bytes`
**cap** in `load_open_text`, not a corpus: `data/tinystories_20k.txt` is
`18,167,706` bytes, `18.17M` tokens, `3.69×` less. And attained peak **drifted
20%** between two runs on this card (bf16 `27.349 → 31.762` TFLOPS) while
*achieved* reproduced to `3%` (`13.575` against `13.205`). **Hours must be
computed from measured achieved FLOP/s, never from MFU × attained peak** — MFU is
the unstable factor here, which qualifies every MFU figure on this page.

### The head dimension is a 31.8× cliff

A 2× throughput discrepancy between two lanes was attributed to bf16 autocast.
It is not. Holding the parameter count identical at **12,281,280** and
`d_model = 352` fixed, moving only `n_heads` so that `d_head` goes 44 → 32:

| `d_head` | bf16 / fp32 | SDPA median | bf16 peak memory |
|---|---|---|---|
| 44 | **0.943×** | 14.753 ms | 4,750 MiB |
| 32 | **2.953×** | 0.464 ms | 1,879 MiB |

At `d_head = 44` both fused backends are rejected — `can_use_flash_attention`
False, `can_use_efficient_attention` False — and the math kernel runs,
materialising `[B, H, S, S]`. **bf16 autocast is a genuine 2.95×, and it pays
only when the head dimension is a multiple of 8.** An earlier MFU reading of
`0.1208` that appeared to fail a `0.20` gate was this artefact; at the project's
own default shape (`d_model = 512`, `d_head = 64`) measured MFU is `0.496`.

### The model does train on tokens

Confirmed by running with instrumented counters rather than by reading
docstrings. All three reachable operators forward to a finite loss, backward with
every trainable parameter receiving a finite non-zero gradient, and every
parameter moves after one step — `25/25`, `25/25`, `39/39`, the last including
`ArmSMPrime`'s own `beta`, `qk`, `g`, `m_head` and `theta_head`.
`torch.nn.functional.scaled_dot_product_attention` was monkeypatched with a
counter and read **exactly 0** across every forward, while the operator's own
counter read `n_layers`, so the operator path is genuinely exercised and never
falls through to stock attention.

The vocabulary path is byte-level, `vocab_size = 256`, UTF-8 bytes in
`ceq/hf/train.py::ByteBatches` and `ceq/lm.py`. There is no BPE or SentencePiece
tokenizer anywhere in the tree, and a search for `*.spm`, `*tokenizer*.json`,
`*vocab*.json`, `*merges*.txt` and `*.model` returns no hits. That is a design
fact, not a defect, and it is recorded so nobody looks for a tokenizer that was
never there.

---

## 5. No checkpoint has a gate

Every `results/**/*.pt` was loaded with `weights_only=True` and the union of
weight-key names taken across all of them. **The union is eight names:**

```
mlp.0.bias  mlp.0.weight  mlp.2.bias  mlp.2.weight
readout.bias  readout.weight  wk.weight  wq.weight
```

158 of 165 files carry a `state_dict`, and every one of those is
`kind='softmax'`, `beta=0.5`. There is no `m_head`, no `theta_head`, no `log_m`,
no gate parameter of any kind, in any checkpoint in this repository.

The gate is implemented and it trains — that is measured in §4. What does not
exist is a single *saved* run that used it. Every trained artifact this project
can point at is the corner the operator was built to improve on.

This is what blocks the three live claims at once. The cost argument needs a
trained `m̄` to know the bandwidth. The refusal argument needs a trained `m̄` to
know where the underflow window sits. The architecture argument needs a trained
gate to have anything to score. One gated checkpoint, trained and saved, converts
all three from argument into measurement, and nothing else in this phase does.

---

## Limits

The calibration bed is synthetic: a 28-state chain with a known transition
matrix, not a decision stream with realised outcomes. Every reliability and
resolution number on this page is therefore a property of that bed, and none of
them has been measured on anything this operator would be deployed against.

Murphy 1973 is cited from its decomposition as reproduced numerically here; the
paper itself was **UNREACHED** across 11 fetch attempts behind an AMS paywall,
and the identity is verified rather than quoted. The claim that ECE requires a
debiased estimator was checked and is **not** licensed — debiasing is one option
among several.

The cost table was measured on a laptop RTX 4060, which is not a training device;
it transfers only by ratio of attained peaks, and the 20% attained-peak drift
recorded in §4 applies to that ratio too. The degree law in §3 is exact for the
scalar resolvent `f(x) = 1/(1 − γx)` and says nothing about a matrix-valued read
whose bandwidth is not known.

The `seg` route in §2 was bound against a hard-clamped gate. Its sharpest
untested attack is a bfloat16 gate value that *rounds* to `0.0` without the clamp
firing, which would make `b_k` fire spuriously and hand the route the disease it
was built to cure.

Phase H's §8 positive retains the confound recorded there: arms (c) and (d) have
a forward pass that is character-for-character the bed's own generative
recursion. Its producer chain was landed at `tests/foreman/h1/` during this phase
and re-runs in six commands, which makes the number reproducible without making
it unconfounded.
