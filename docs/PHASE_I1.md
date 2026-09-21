# Phase I.1 — Repair

Every gate number in this repository was taken from a start where half to 98% of
the gate was closed by an initialiser nobody chose, on a corpus 14–28× too small
for the models, on a torch build that never touched the card. This phase repairs
the instrument before any further number.

The rows below land as they are measured. Rows still out are named in §4.

---

## 1. The clamp is the freeze

`ceq/arm_smprime.py`'s `magnitude` is `clamp(u, 0, 1)`, whose gradient is **zero
outside the open interval**. On LayerNorm-normalised rows at `d = 512`, weights
`N(0, 1/d)`, 20,000 samples, float64:

| start | at exact 0 | at exact 1 | carrying gradient |
|---|---|---|---|
| bias 0 | **0.501** | 0.160 | **33.8%** |
| bias 0.999 | — | **0.498** | **34.3%** |

The bias gradient is a mean over per-gate gradients, so with two thirds of the
gates dead at *either* start it is small in both directions. That is why two
training runs ended with `m_head.bias` within `0.02` of wherever it began — from
`0` to `−0.0172 / −0.0021 / +0.0068 / −0.0101`, and from `0.999` to
`0.9797 / 0.9987 / 0.9908 / 0.9990`. **"Frozen at initialisation" is the clamp,
not the data.**

And the sharper consequence, for the refusal channel this project is built on: a
gate that reaches exactly zero under a clamp has no gradient to leave by, so it
**never reopens**. It is dead rather than closed, and a refusal that cannot be
revised is a different object from a refusal the model chose.

Every gate row in `docs/PHASE_I.md`, `tests/foreman/gated/` and
`tests/chase/cogs/` carries a provisional banner until re-measured under a
parameterisation that keeps exact endpoints reachable with a live gradient.

---

## 2. The corpus that ends the memorization regime

The regime was never labelled, and every language-model comparison this project
has made sits inside it. `data/tinystories_20k.txt` is 18,167,706 byte-level
tokens; at 20 tokens per parameter the zero-repetition ceiling is **908,385
parameters**. The measured configurations need far more:

| row | params | tokens at 20/param | repetition on the 18.2M file |
|---|---|---|---|
| 12M | 13,107,200 | 262,144,000 | **14.43×** |
| 25M | 25,690,112 | 513,802,240 | **28.28×** |

**A gate that helps or hurts memorization says nothing about generalization.**

### The decision: full TinyStories, and it was derived rather than recalled

`data/README.md` records `tinystories_20k.txt` as a head of the TinyStories train
split at 20,000 rows. `wc -c` gives `18,167,706` bytes, so `908.39 B/row`. The
fetched train split is `2,141,709` rows, putting the full split at
`2,141,709 × 908.39 = 1.945e9` bytes.

| row | repetition on full TinyStories |
|---|---|
| 12M | **0.135×** |
| 25M | **0.264×** |

Both below 1×. The sensitivity is stated rather than assumed: the 25M row fails
only below 565,617 rows, so the fetched count would have to be overstated
**3.79×**; at half the head's mean row size it still clears by 1.9×. Licence
`cdla-sharing-1.0`.

**It stays `[U]` — no digest, no revision hash — so the pin is enforced as a gate
on the run rather than a promise.** The training lane prints `bytes_total`,
`tokens` and `repetition = tokens_needed / bytes_total` beside the loss, and
labels the row **MEMORIZATION REGIME** whenever `repetition ≥ 1`.

Rejected with reasons. **WikiText-103** (191.98 MB, CC BY-SA 4.0) clears neither
budget, and pairing a second corpus buys a genre mixture no existing row used.
**FineWeb-Edu** `sample-10BT` (~10B GPT-2 tokens, `odc-by`) clears both by an
order of magnitude but is **UNREACHED** on the specific question of
unauthenticated reachability from this box, and is the largest domain shift — it
asks whether the operator generalizes to web text, which is a different
experiment rather than a continuation.

If the pin fails, nothing reachable in-domain clears 514M tokens, and the honest
move is the certificate programme below on the file that exists.

### The certificate shape

```
d_model=128, n_heads=8 (d_head=16), n_layers=3, seq=512, vocab=256
-> 720,896 params, 79.4% of the 908,385 ceiling, repetition 0.79x
```

`P = 12·L·d² + 2·vocab·d + seq·d`, reverse-derived and confirmed exact on both
existing shapes (`d=512, L=8 → 25,690,112`; `d=512, L=4 → 13,107,200`).
Per-layer LayerNorm parameters add `4·L·d = 1,536`, so the real count is
`~722,432`, still under the ceiling.

`n_heads` is **8, not 16**, at identical parameter count — attention is `d²`
regardless of head count, `d_head = 16` still satisfies the multiple-of-8 rule,
and 8 matches both existing shapes, so the certificate row changes `d_model` and
`n_layers` and nothing else. Depth is maximal: `d=128, L=4` costs 917,504, over
the ceiling; the only competitor is `d=160, L=2` at 778,240, and `L=3` beats
`L=2` for a project whose claim is about composition.

**This is the first language-model row in the project entitled to a
generalization sentence.**

---

## 3. The audit: four kills, and a search that could not find them

The round's known process failure was one lane's checker running `taskkill /F` on
two GPU PIDs and destroying a sibling lane's training run. The audit found three
more, and struck its own first pass on the way.

**The ground pass reported zero occurrences of `taskkill` in the board log.** The
incident *was* logged — `house-events.jsonl:4084`, agent Cameron, kind
`incident`, severity high. The grep searched for `taskkill` against a line that
says `Killed PIDs`. **A search with no true-positive path: a check that cannot
fail, found inside the audit for checks that cannot fail.**

**And the session's own watchdog reported three kills that did not happen.**
`scratchpad/deadline_kill.sh` ran `Stop-Process -Force -ErrorAction
SilentlyContinue`, swallowed the refusal, counted the attempt, and printed
`killed $n process(es)`. PID 9000 — `python -m ceqjepa.hbucket`, started
2026-09-20 13:28:47 — was reported killed three times and is **still running**.
Repaired: the loop now verifies the process is gone before counting it, reports
`attempted` and `verified killed` separately, and logs `REFUSED` with the PID
when `Stop-Process -Force` does not take.

### The standing catalogue of checks that cannot fail

| # | check | why it could not fail |
|---|---|---|
| 1 | load-bearing gate control | added a **constant** to every `m_head` weight; `m_head` consumes a LayerNorm output whose row sum is zero, so the shift lay **exactly in the null space**. Measured `3.5e-6`; a random perturbation moves logits `0.716` |
| 2 | control reproduction | compared against `0.02930` at `1e-9` tolerance; the on-disk value is `0.029296875`, so a bitwise-perfect reproduction differs by `3.1e-6` |
| 3 | generalization reproduction | `\|0 − 0.029297\| = 0.0293 < 0.03`, so a model scoring **exactly zero** passes |
| 4 | memory-wall detector | caught only `torch.cuda.OutOfMemoryError`, so it could not see a host spill that halves throughput while raising nothing |
| 5 | `tests/beds/test_target_pin.py` | two construction checks — a threshold on a bed-constant tensor, and `trained_std ≥ frozen_std` on the literal same bitwise tensor (`x ≥ x`). Self-corrected by the project before this sweep reached it |
| 6 | kill audit grep | searched `taskkill` against text reading `Killed PIDs` |
| 7 | watchdog kill count | counted attempts, not verified deaths |

| 8 | straight-through forward check | compared `torch.clamp` to `torch.clamp`; could not detect a changed forward |
| 9 | pairing's shuffled control | vacuous as constructed; pairing held on other evidence |
| 10 | Kaggle notebook's rebuild gate | numpy-only; executed none of `magnitude_clamp`, `path_product`, `hop`, `operator`, `readout` or `GatedBlock`, so it would pass with the whole torch rebuild wrong |
| 11 | span-containment curve | tracks `1 − (1−p)^L` to three decimals, so it measures the density it was handed rather than a mechanism |

**Open.** Roughly 140 further tolerance assertions across `tests/cameron`,
`tests/curvature`, `tests/foreman` and `tests/lorasort` were located by the same
sweep and **not** individually verified; five were spot-checked and none was
decorative. That is recorded as an open sweep item rather than as a clean bill.

---

## 4. Second signatures

**Verdict 1 — the operator represents non-commutative composition. Signed, with
scope.** I-AUT with the generator handed over as bare integers: operator `0.8620
± 0.0556` against a commuting-diagonal control at `0.2860 ± 0.0150`, matched at
404 parameters by `numel()`, the control saturated within `0.031` of its own
multiset ceiling of `0.3110`. The scope that belongs in the signature: **one
parameter budget, one word length, no length-generalisation check.**

**Verdict 2 — the operator predicts. Not signed.** Recalibrated resolution
`0.001469` against an oracle ceiling of `0.10117` is **1.45% of the resolution
the bed offers**. The contract's re-reading is endorsed: this is **not a tie at
the ceiling, it is both arms at the floor**, and the row's verdict is
*undertrained*. Its next measurement is a size sweep, not a new architecture.

---

## 5. The guards, and three contract numbers that do not hold here

Six guards built; five fire when an independent checker triggers them rather than
reading a report. Every proposed change is a diff in `r5_guards.md`; none is
applied to `ceq/`.

| guard | fires | silent |
|---|---|---|
| head dim `% 8` | `d_head` 44, 33, 12 | 32, 64 |
| forced-fused SDPA | `d_head=44` bf16 — `can_use_efficient_attention` False, verified live | 64 bf16; 32 and 64 fp32 |
| host spill | L=9: peak `8578.1 MiB` against a `8187.5 MiB` card, **no `OutOfMemoryError`**, status `ran` | L=6, L=7 |
| depth default | refuses L=7 without `acknowledge_headroom=True`, hard-refuses L=8 even with it | L=6 |
| dtype refusal | bf16 and fp16 at construction, naming `torch.polar` and `_ctype` | fp32; `sgate`; `signed` |

The dtype guard raises at construction through an `_apply` override, because HF
modules are **cast after construction** rather than given a dtype at `__init__` —
a `__init__` check would never see the dtype that crashes. The two crashes it
replaces both reproduce verbatim without it: `double != struct c10::Half` for
fp16, and `Expected both inputs to be Half, Float or Double ... but got BFloat16`
for bf16. A runtime spy confirms the shipped `gate()` calls `torch.polar` while
the three-channel path never does.

**The host-spill guard is signed at L=1–4 only.** The checker could not
re-observe the L=9 fire: the card sat at `7837/8188 MiB` under two sibling lanes
and the runs starved past 25 minutes. **He killed nothing and waited**, which is
the correct behaviour under the standing prohibition and is recorded rather than
worked around. The L=9 fire therefore rests on one agent's observation and is
labelled as such.

### Three numbers from the contract did not reproduce

**The head-dim penalty is a bfloat16 defect, not an fp32 one.** At fp32,
`d_head=44` is **1.41× slower, not 31.8×**, because memory-efficient attention
*accepts* it on this box. The reproducing case is **bf16 `d_head=44` at 12.22×**,
peak `183.4 MiB` against `16.1`. And **flash attention is never compiled into this
Windows build at any shape or dtype**, so no guard can assert its presence. The
substitution was made rather than the contract's figure reported as a measurement.

**The `exp(S·5.2e-4) − 1` rounding law is wrong in functional form.** Re-measured
relative error is `0.079524 / 0.281715 / 0.734220 / 0.995026` at
S = 64 / 256 / 1024 / 4096 — it **saturates toward 1.0** rather than growing to
`7.41`. The S=4096 point matches the independently recorded `gate3.md` Cell A
figure of `0.99510` to four significant figures, which is why the re-measurement
is trusted over the formula.

**And the crossover is S = 75, not S = 256.** The first pass reported 256 as "the
first S over the 0.094 bound", but that was an artefact of testing only four
powers of four. Bisected, the bound is crossed at **S = 75**.

That settles R6. A bf16 mantissa fails below any sequence length this project
runs, so the three-channel path is **fp32-mantissa permanently** — `torch.polar`
still leaves the hot path, and no complex tensor is created, but the bf16 saving
is not available.

### The memory law reproduced

Re-fit live from L=6, 7 and 9: `peak_mib = 929.96·L + 208.56`,
R² `0.9999999458` — within `0.06 MiB` per layer of the contract's
`929.9·L + 208.3`. Default depth **L = 6**; L = 7 by explicit override; L = 8
refused, which changes `ceq/hf/train.py::DEFAULTS` from 8 to 6.

One incidental defect, fixed in place and reported rather than patched over: the
run's own board logging raised on a `numpy.bool_` that `json.dumps` refuses.

---

## 6. The eval path, and what the determinism null did not test

**R3 is closed.** `ceq/hf/train.py` had no held-out quantity of any kind —
`grep -ic eval` returned 0, `ByteBatches.val` was built and never requested. It
now splits **by document**, 105,095 blank-line-separated stories seeded through
`split_seed`, giving 94,585 train and 10,510 val with **zero document overlap**,
and writes `eval_losses` to `run_record.json` alongside `val_frac` and
`split_seed`. Proposed as a diff; not applied.

**The split was checked by searching for the leak rather than by reading the
code.** Held-out documents were searched against the concatenated train split:
**0 of 96** mid-document 200-character windows appear in train. Two of 200 whole
held-out documents match verbatim and both are corpus fragments — `"The end."` at
8 characters and `"Are you OK?" Lily asked.` at 25 — not stories. The val split is
not a contiguous tail, and `split_seed` 0 against 1 produces different val md5s,
so the seed genuinely redraws.

**And the leak it closes is 56 bytes.** The stock byte-offset cut at 16,160,346
does land 114 bytes into a 170-byte story exactly as predicted, so 56 bytes of one
story crossed into val — **0.0031% of the 1,795,595-byte val split**. Corpus-wide
there are 704 duplicate document copies, 0.67%, of which only 2 are 100 characters
or longer, so duplication is not a second leak channel. The by-document split is
correct and the defect it closes was small; both are stated.

**The seed bug is fixed and the fix is demonstrated.** `eval_indices(3920, 256,
seed=1)` and `seed=2` returned byte-identical index lists against the live
unedited function. Folding the seed into the formula —
`manual_seed(20260825 + n_test + seed)` — makes 1 and 2 diverge while keeping
**seed 0 byte-identical**, so every existing seed-0 result, including the COGS
control curve, is unmoved.

### The determinism null is a null about the wrong model

Six runs — two without the flags, two with, two for the must-fire — returned
**bit-identical** eval-loss trajectories, final `2.249588042497635` in every one.
The flags cost 7.6% peak memory and removed nothing.

That reads as "determinism was already free", and it is not what was measured.
The `0.113` swing this row exists to explain lives in `ceq.lm`'s literal
scaled-dot-product attention path, and `CEQForCausalLM` has **zero
`scaled_dot_product_attention` call sites**. The model used here cannot exhibit
the mechanism. **The 0.113 swing is untested, not unremoved**, and the R5 fused-
backend assertion is likewise not applicable to this module for the same reason.

The must-fire needs the same correction: its band was set from repeat-determinism
variance, because **no seed was varied across the six runs**. It measures that one
seed reproduces itself, which it does exactly, rather than that eval loss
reproduces inside seed variance. The threshold was set before the confirmation
pair was inspected, so the procedure was sound; the quantity was the wrong one.

Every row here is labelled up front rather than caveated afterwards:
`n_params = 4,929,536` against the 908,385 ceiling gives a repetition factor of
**5.4267×**, **MEMORIZATION REGIME**, and no generalization sentence is made about
any of them.

**Kaggle's blocker is now closed**: a remote run has a held-out quantity to score.
What it does not yet have is a seed-variance estimate on a model that can vary.

---

## 7. R1: the kill fires, and exactness becomes eval-only

The pre-registered kill was written before any number: *if both parameterisations
fail the must-fire, the closed-magnitude gate cannot be trained with live
gradients at exact endpoints; exactness becomes eval-only and the page says so.*

**Both fail.**

| | (i) live gradient ≥ 0.95 | (ii) bias moves ≥ 0.05 | (iii) init zeros < 0.05 |
|---|---|---|---|
| straight-through | PASS `1.0000 / 1.0000` | **FAIL `0.01923`** | PASS `0.01825` |
| hard-concrete | PASS `0.9978 / 0.9863` | **FAIL `0.02757`** | PASS `0.00000` |

Both move `m_head.bias` **inside the same `~0.02` band the zero-gradient clamp
itself moved in**. Giving every gate a live gradient did not unfreeze the bias.

**Exactness is therefore eval-only: train with hard-concrete, freeze to clamp at
eval.** No form is set as the default, because the bar that was written first does
not endorse either as trainable-with-live-gradient.

The dead-zone diagnosis reproduces exactly and is not a seed artefact.
At bias 0, `P(u ≤ 0) = 0.5014`, `P(u ≥ 1) = 0.1602`, `33.84%` carrying gradient;
at bias 0.999, `P(u ≥ 1) = 0.4982`, `34.33%`. Independently re-derived, matched
against the closed form `Φ(1) − Φ(0) = 0.34134` and
`Φ(0.001) − Φ(−0.999) = 0.34150`, and stable to swapping the draw order
(`0.50135 / 0.16125`). Hard-concrete at bias 0 puts `0.855%` at exact zero and
`0.860%` at exact one.

### The result the must-fire does not capture

| | last-50 loss | trained exact-zero | backward reach mean / median / max |
|---|---|---|---|
| repaired clamp | `1.2665 ± 0.0399` | `0.3121` | `2.146 / 1 / 42` |
| straight-through | `1.2999 ± 0.0439` | `0.4050` | `1.209 / 1 / 13` |
| **hard-concrete** | **`1.1391 ± 0.0401`** | **`0.0006`** | **`241.4 / 234 / 512`** |

Trained exact-zero fractions are reported with no threshold, as condition (iv)
requires.

**Straight-through gives every gate an unbounded live gradient and the gate still
closes — to `0.4050`, worse than the `0.3121` of the clamp it was built to
rescue, and to a loss worse than that clamp's.** So the collapse is *not* only the
dead zone. Something in the objective prefers the gate shut, and that is a
different question from the one this row answered.

**Hard-concrete ends at a median backward reach of 234 at sequence length 512**,
against `1` for both other forms, at the lowest loss of the three.

**It does not achieve that reach — it preserves it.** Measured at
initialisation, hard-concrete already reads `256.5` and the other two read median
`52`, mean `79.07`. So training moves hard-concrete `256.5 → 234` and moves the
other two `52 → 1`. The honest statement is that **two forms destroy their initial
reach and one does not**, which is a different claim from the one an earlier draft
of this section made.

### The zero-density account was tested and is dead

A leap proposed that the whole effect is the **density of exact zeros at step 0**,
made permanent because `∂G_ij/∂m_k = ∏_{l≠k} m_l` — one zero in a span silencing
every gate in it. A crossed design was built to separate form from density, and
the account died before either cell finished, on two independent grounds.

**It explains a table it is not about.** All three rows above ran at
`m_head.bias = 0.999` (`tests/chase/gate/r1_gate.py:260`), whose measured init
zero densities are `0.018250 / 0.018250 / 0.000000` — **not 0.50**. The premise
"at bias 0 the clamp puts a zero every second token" describes the **pre-repair**
init, measured here at `0.491455`, and the pre-repair run is not in this table.

**And the table refutes it without new training.** Straight-through's forward is
*bitwise* clamp, so at seed 0 it carries the identical init density `0.018250`,
the identical span curve and the identical init reach `52 / 79.07` as the repaired
clamp — then ends at a different trained zero fraction (`0.4050` against
`0.3121`), a different max reach (`13` against `42`) and a different loss, off a
**bitwise-identical step-0 loss**. Same step-0 density, different endpoint, so
step-0 density is not the effect.

**The crossed cell was also void by construction.** Clamp at `bias = +3.0` does
reach zero density `0.0` — with `m == 1.0` at **100% of positions** and
`frac_grad_nonzero = 0.0000`, pinning every gate at the *other* saturating
endpoint and removing all decay from the path product. At the measured
`u_std = 0.443641` the clamp's live interval `(0,1)` is **2.254σ** wide, so the
lowest bias reaching hard-concrete's density is `1.336`, where **77.6% of gates
are already dead at m = 1**. No clamp cell holds that density with a live
gradient at this init scale.

**The span curve is arithmetic, not evidence.** It tracks `1 − (1−p)^L` to three
decimals, so it measures the density it was handed. At the density the table
actually ran, `L8 = 0.088` and `L32 = 0.302` — **70% of length-32 spans carry no
zero at init**. And the clamp row's zeros **grew** under training,
`0.018250 → 0.3121`, so they are the consequence rather than the cause.

**What survives.** The R1 kill stands on its own measurement. Why the gate closes
is open, and the two accounts offered so far — gradient starvation, and step-0
zero density — are both refuted by the same table.

### R7 is clean, and the instrument was validated before it was trusted

**Zero spurious zeros on every measured cell** — both forms, at init and trained —
via flag and cumsum against the float path product. The instrument was self-tested
on a synthetic `S=16` case with one true zero (`n_true_zero = 55`, exact), then
cross-validated against this project's own recorded defect: constant `m = 0.5`,
`S = 4096`, float64, **measured 4,564,731 spurious zeros, an exact match** to the
recorded count, with `S = 64` float64 correctly reading 0.

### Two more checks that could not fail

The race's straight-through forward-identity check **compared `torch.clamp` to
`torch.clamp`** — it could not have detected a changed forward. Forward identity
is nonetheless established by a different route: the init exact-zero fraction
`0.01824951171875` is bit-identical to the repaired-clamp reference's `299/16384`,
and `loss_first = 5.670821666717529` is bitwise identical to that run's.

And the shuffled control that was supposed to prove pairing is **vacuous**.
Pairing holds on stronger evidence — a bitwise-identical step-0 loss against the
reference — with the detrended correlation at `0.9621` against `−0.0298`.

The `(ii)` failure was confirmed on **all four layers read out of the
safetensors**, not the single scalar the race reported.

---

## 8. Rows still out

`R1` gate parameterization (straight-through against hard-concrete, with the
four-condition must-fire); `R3` held-out eval path by document and `R4`
determinism; `R5` hardware assertions and `R6` dtype guards, each of which must
be demonstrated firing rather than described.

**Kaggle remains closed** on the verified absence of any held-out quantity, until
`R3` lands.
