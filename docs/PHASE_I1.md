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
| 11 | freeze verifier | compared `clamp(u_raw,0,1)` while `blend()` computes `magnitude(lerp(1,u,g))` with `g` trainable and drifted to `0.8699`; halted a row on a false alarm |
| 12 | span-containment curve | tracks `1 − (1−p)^L` to three decimals, so it measures the density it was handed rather than a mechanism |
| 13 | C27 rebuild gate anchor | specified against `ceq/arm_pl.py` and `ceq/arm_phase.py`, which contain neither `path_product` nor `hop` — a gate pointed at a file without the functions it compares |
| 14 | C25 check-the-check | read ORDERING UNSTABLE from `sorted()`'s alphabetical tie-break between two arms both scoring `RES = 0.000000` |
| 15 | `score_vs_ceiling()` | the matched-functional enforcer, defined and never called; the ceiling script prints a binned numerator over an unbinned published value |
| 16 | band-position step 0 | tested `ρ(P)` for a row-stochastic `P`, which is `1.0` by Perron–Frobenius whatever the weights — the kill could only ever fire, and fired without consulting a trained parameter |

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

**Exactness was therefore prescribed as eval-only: train with hard-concrete,
freeze to clamp at eval.** No form is set as the default, because the bar written
first endorses neither as trainable-with-live-gradient.

> **That prescription is dead. See §9.** Freezing to clamp at eval is not a
> freeze — it is a substitution of a different function, and it reconstructs the
> defect it was written to escape.

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

## 8. S1: the kill fires on the point, and its reading does not survive the floor

`RES/ceiling` at ~200k parameters reads `0.02495` at **both** seeds, below the
pre-registered `0.05`, so the encoding kill fires and replicates. Three things
stop it meaning what it was written to mean.

**The interval contains the bar.** `[0.00415, 0.07521]` of ceiling at seed 0,
`[0.00436, 0.08228]` at seed 1. It fires on the point estimate, not at CI.

**The scorer and the ceiling are different functionals.** Resolution is scored
over **10 fixed-width one-vs-rest bins**; the ceiling is an **unbinned per-class
variance**. `white_win` and `black_win` contribute `res_k` of **exactly 0** at
both seeds, because every recalibrated forecast lands in bin 0 — the scorer is
blind to rare-class resolution while the ceiling counts all four classes.

**And re-binning inverts the ordering.** At tier 2, seed 0:

| | fixed-width (10) | equal-count (10 / 200) |
|---|---|---|
| operator | 0.002155 | 0.003769 |
| twin | 0.001102 | 0.001721 |
| **piece-count floor** | 0.006761 | **0.006993** |

Under fixed-width the operator sits `2.3×` **above** the floor; under equal-count
it sits `3.1×` **below** it, and the floor reaches `0.069` of ceiling — **above
the 0.05 bar the operator fails**. A strictly-less-informative 8-bin summary of
the operator's own input clears the bar the operator misses. **That is the model
failing to reach what its own encoding already supports, not the encoding being
the ceiling, and `fen_to_vec` must not be touched on this evidence.**

Three further claims of the sweep do not hold. `RES/ceiling` **does not rise with
size**: at seed 0 the operator rises and the twin falls, at seed 1 both reverse,
and every per-arm trend direction flips between seeds while each tier's CI
contains the others' points. The arms **separate at zero of three sizes**, and
the operator-minus-twin gap **sign-flips at all three tiers** between seeds. And
"trained to convergence" means the early-stop became eligible only past
`0.8 × budget`, with all twelve runs stopping within ~180 steps of eligibility —
`converged=True` records that the budget nearly ran out. At tier 1 the operator's
final `L_q` is `0.985` against the twin's `0.082` on the identical objective.

Scope, stated: 300 games of the pre-registered 5,900, two seeds, and a 2k tier
that is **architecturally unreachable** — `x_dim = 769` floors `TinyCEQ` at 8,289
parameters.

---

## 9. S2: the eval-only prescription is dead

R1 prescribed training with hard-concrete and freezing to clamp at eval. Scored
on COGS, that arm reads **in-distribution `0.0000`, 0 of 256**, against
`RESOLUTION_FLOOR = 0.20` — **below the broken bare-clamp arm's own `0.16797`**.
It fails the admission gate, so the row stops and no generalization curve exists.

**The same weights, scored under the training form, read `0.3125` and clear the
floor.** One set of parameters, two readouts:

| readout | annihilated | live keys / 192 | median backward reach |
|---|---|---|---|
| frozen clamp | **98.3579%** | **1.5846** | 2 |
| hard-concrete | **0.0%** | **96.5** | 92 |

Training drives `u` below zero, where hard-concrete still passes signal
(exact-zero fraction `0.0011`) and the clamp annihilates (`0.3490`). **So "freeze
to clamp at eval" is not a freeze — it is a substitution of a different
function**, and it reconstructs the broken arm's `97.99% / 1.93` almost exactly.

The arm remains a milder underfit — train loss `0.3494` against the control's
`0.2731` at identical budget — and that is **not** the cause of the zero, since
the same weights score `0.3125` when read with the function they were trained
under.

**The freeze verification was itself defective, and predicted before the run
reached step 3000.** `blend()` computes `magnitude(lerp(ones, u, g))`
(`ceq/arm_smprime.py:134`) while the verifier compared against
`clamp(u_raw, 0, 1)`; `g` is a trainable `nn.Parameter` (line 725) inside AdamW
and had drifted to `0.8699419498443604`. The check read `freeze_took = false` at
max difference `0.1299` and **halted the row on a false alarm**. Compared on the
blended argument instead: `bitwise_equal_to_clamp_forward = true`, max difference
`0.0` over n = 768 captured from a real eval forward, and
`differs_from_hardconcrete_forward = true` at `0.49994`. The freeze is real; the
instrument was not.

One seed. On a bed with a measured `0.113` fixed-seed swing, that is a data point.

---

## 10. Kaggle: the seed-variance interval that did not exist

Run on a **Tesla T4**, `melowdramtic/kg-wide-breadth-row`, self-contained — no
repository file, no project data, no token. Eight seeds of the SDPA-literal arm,
the only path in this project carrying a real `scaled_dot_product_attention` call
site, which is why the local determinism null could not see anything:

```
finals  2.2976 2.2973 2.2973 2.3029 2.2981 2.2982 2.2981 2.3010
swing   0.005563836097717267
stdev   0.001885094358229673
```

**Across-seed swing is `0.0056`.** The COGS figure quoted throughout this phase is
`0.113` — **twenty times larger, at a fixed seed.**

The two measure different things and the comparison is suggestive rather than
decisive: this is *different* seeds on a byte-level language model scored by loss,
while the `0.113` was the *same* seed four times on COGS scored by exact match.
But it locates the problem. If changing the seed entirely moves this model by
`0.0056`, then a `0.113` fixed-seed swing is not ordinary kernel
nondeterminism. At 512 evaluation items with a quantization floor of
`1/512 = 0.00195`, `0.113` is **58 items flipping** — which points at the
evaluation subsample and the exact-match scorer rather than at training, and the
local sweep looked at training.

**The session then ran out of memory** — `512.00 MiB` requested against
`424.81 MiB` free on a 14.56 GiB card — before Q2, the certificate row, and Q3,
the gate sweep at breadth. Incremental writes meant the tail was lost rather than
the run: 33 rows landed and Q1 completed.

Three defects were repaired before the push, and the first would have inverted the
finding the row exists for. The notebook computed
`repetition = tokens_seen / (20·n_params)` — the **reciprocal** of the row's
definition, with the wrong numerator — so a **larger** model scored a **lower**
factor and would have been labelled generalization-admissible. Corrected to
`(20·n_params) / corpus_tokens` and checked against the row's own four values:
`0.7936 / 0.6963 / 14.43 / 28.28`. `torch.manual_seed` ran **after** model
construction at every call site, so a logged seed reproduced the batch order and
not the initialisation — inside the seed-variance question itself. And the
certificate shape is **632,496** parameters rather than the 720,896 the closed
form predicted, a 12.3% shortfall moving its repetition factor to `0.696`.

---

## 11. C26: the floors, and two figures that do not clear them

Every exact-match number carries `n` and its quantization floor `1/n`, and **a
swing smaller than `3/n` is not a swing.** Fourteen published figures were swept
against that rule. Twelve survive. Two do not, and both are new.

**`cogs_curve.jsonl`, in-distribution, step 9,000 → 12,000: the delta is
`0.00390625` — exactly `1/256`, one item — against a floor of `3/256 = 0.01172`.**
The apparent movement in that leg of the curve is subsample noise. The curve's
generalization column is unaffected: its deltas are `0.11523`, `0.00977` and
`−0.04883` at `n = 512`, all clear of `3/512 = 0.00586`.

**`addprim_jump`, softmax, per-seed swing `0.005859375` — sitting exactly on the
`3/512` boundary.** It is the smallest value the rule admits as a swing at all,
and it has been read as neither signal nor noise anywhere.

The known instance reproduces: the COGS `0.113` at `n = 512` is 58 items against
a `0.00586` floor, and survives comfortably.

### The subsample is retired by a flag, not a diff

`ceq/harness.py::eval_indices` already returns `range(n_test)` whenever
`max_eval ≥ n_test`, and `ceq/capability.py` already exposes `--max-eval`. So

```bash
python -m ceq.capability --split cogs --max-eval 21000
```

scores the **full** 21,000-item generalization split today. Priced from the four
measured `eval_seconds` in `cogs_curve.jsonl` — a per-item range of
`0.00951`–`0.04570` s — that is **3.3 to 16.0 minutes per checkpoint**, against a
training wall clock of `1737.9` s already recorded for the same split. The
subsample costs more in credibility than it saves in time.

### PID 9000 is defunct, and the reboot clause does not apply

`Get-Process -Id 9000` fails — *"Cannot find a process with the process
identifier 9000"* — while `tasklist` and `Get-CimInstance Win32_Process` both
still list it as `python.exe -m ceqjepa.hbucket`, created 2026-09-20 13:28:47.
**That three-way split is the mechanism behind `taskkill`'s "no running instance"
against a PID `tasklist` still prints.** `nvidia-smi` reads `0 MiB / 8188 MiB`
with no running compute processes.

**The memory is not held.** An earlier reading of 1,779 MiB at 95% utilisation
was the sibling lanes, not this entry, and the claim that a day-old orphan was
starving the box is withdrawn. The contract clause *not released means reboot
before any timing row* does not apply; no reboot was performed or proposed.

### The rebuild gate was anchored to the wrong file

C27's torch-executing rebuild gate — the check that exists precisely to catch a
wrong rebuild — was specified against `ceq/arm_pl.py` and `ceq/arm_phase.py`,
**neither of which contains `path_product` or `hop`.** The correct anchor is
`ceq/arm_smprime.py`, and the gate was rewritten and run against it. A gate
pointed at a file without the functions it compares is the thirteenth entry in
the catalogue below.

---

## 12. C25: the rule holds, and the first implementation of it broke the rule

A scorer and its ceiling must be the **same functional**. A row whose scorer and
ceiling differ is **void, not inverted** — the distinction matters because an
inverted row invites you to pick the flattering binning and a void row does not.
Resolution is computed at equal-count 200, equal-count 50 and fixed-width 10, and
**the ordering of arms must agree across all three or the row carries no
verdict.**

### The wrapper can fire both verdicts — but not by the demonstration offered

A real instability was constructed and fired: a narrow-band arm with
excess-over-null `0.134` that fixed-width-10 ranks **last**, at `res_k` of exactly
`0.000000` with all 8,000 forecasts in one bin, while **both** equal-count schemes
rank it **first**.

The row's own UNSTABLE demonstration did not establish that. At `equal_count_50`
**both arms score `RES = 0.000000` exactly**, and the reported ordering is
`sorted()`'s alphabetical tie-break rather than a measurement; at
`equal_count_200` the winning margin of `3.28182e-05` comes from a class with
**zero populated bins** and a residual of `2.09e-01`. A check-the-check that reads
unstable from an alphabetical tie is the fourteenth entry in the catalogue.

### `score_vs_ceiling()` is dead code

It is defined at `c25_scorer.py:132` and **called from nowhere in the tree.** The
one function whose entire job is to enforce the matched functional never runs.
What the ceiling script actually prints is a **binned numerator over the published
unbinned `0.10116955630126778`**, at a different `n`, on different data — the void
condition, rebuilt inside the fix for it.

### The binned ceiling reproduces the unbinned one

This is the opposite of what was anticipated. The headline that the binned ceiling
**exceeds** the unbinned one by `1.6×`–`2.4×` is **38–49% uncorrected bin-count
bias**; once that floor is subtracted the binned ceiling reproduces the unbinned
one to within 10%. **The correction was already in the file the row imported
from** — `wil_chess400_results.json` records
`shuffle_null_analytic_B_minus_1_UNC_over_N = 0.0006843155567724138`.

So no RES-over-ceiling ratio needs restating on ceiling grounds, and the earlier
warning that they might is withdrawn.

### Two defects that void any verdict the wrapper issues

**A degenerate edge set silently drops every item.** A constant forecast collapses
`torch.unique(edges)` to a single edge, `_score_one_class` loops over `range(0)`,
and all 6,000 items vanish: reliability falls from a correct `0.00637584` at
fixed-width to `0.0`, the residual jumps to `6.38e-03`, and **nothing raises.**
The resulting `RES = 0` is correct by accident — a constant forecast does have
zero resolution — so the ordering survives a bug rather than being produced by the
scorer.

**There is no tie handling.** Two arms with identical resolution are ordered by
dictionary insertion, confirmed by scoring one forecast array under the names
`zzz` and `aaa` and receiving a STABLE verdict on a pure tie. Both fixes are
named: place all items in one bin when the edges degenerate, and return
**ORDERING TIED** rather than a name-sorted ordering.

### And the published rows cannot be re-scored at all

Only aggregates are saved in the tree, not raw per-item forecast arrays, so the
chess400 and tier-2 rows **cannot be re-scored under any other bin scheme.** That
is a fixed-structure defect of exactly the class L-REFLECTOR exists to catch: the
decision about what to persist was made once, never revisited, and it forecloses
every later audit.

One premise carried from the earlier sweep is also misattributed — `white_win` and
`black_win` are **not** zero on the recalibrated operator arm. Two lanes disagree
on that figure and it is recorded as disputed rather than settled.

---

## 13. The L-REFLECTOR audit: 32 tables, and two mechanisms

Every published table in `PHASE_G`, `PHASE_H`, `PHASE_I` and `PHASE_I1`, and the
producers under them, was audited against its own fixed structure — initializer,
parameterization, corpus regime, scorer functional, bin scheme, dtype path, torch
build, eval subsample size — before anything it scored.

**32 tables: 9 VOID, 10 RESTATED, 13 SURVIVE.**

The result that matters is not the count. It is that **seven of the nine voids
trace to exactly two mechanisms.**

| mechanism | tables |
|---|---|
| **scorer and ceiling are different functionals** | chess resolution at 10 fixed-width bins against an unbinned per-class variance (`I1 §8`); frozen readout scored against trained weights (`I1 §9`); the Murphy residual checked against its own inputs (`I §1`); order-free composition scored by L2 on a sphere, where geometry forces chance regardless of order (`H §5`) |
| **the regime forbids the sentence** | every prior LM comparison at 5.4×–28.3× corpus repetition (`I1 §2`); a bar set from one non-monotone checkpoint that the curve crosses in both directions (`I §6`); a density account explaining a table that ran at `0.018250` rather than the `0.50` it assumed (`I1 §7`) |

The remaining two voids are a Fourier reproduction bar that fails its own
pre-registered 150-step threshold by `0.0063` nats (`G §5`), and the eval-only
freeze prescription, which is the same functional mismatch appearing a second
time on a different bed.

### Both live verdicts survive their own fixed structure

**The I-AUT confound-removed race SURVIVES** — the pre-registered kill against the
commuting-diagonal control did not fire, 5 of 5 seeds pairwise, no overlap of the
spreads, on a generator handed over as bare integers.

**The chess ceiling at `max_plies` 80 → 400 SURVIVES** — the same oracle-resolution
functional computes both the ceiling and the arm, and the intervals are
non-overlapping at 202×.

**The committor row is RESTATED, not void.** It beats a base rate and loses to a
summary of its own input, and the correct sentence is *undertrained*, not *not
predictive*.

Ten further tables are restated rather than struck, including the head-dim cliff
— corrected from `31.8×` to `12.22×` and scoped to bfloat16 only — and the
host-spill guard, downgraded to signed at depths 1 through 4 only.

### The hop gate, and it discriminates

Both exemplars that beat this project's own leap seat share a shape: **audit the
object that cannot move under the model's own rules — the reflector's involution,
the S-matrix's unitarity — then treat what survives elimination as the answer.**
A structural NEVER is not a refusal; it is what the Bombe searched on.

Five conditions, and a leap is not dispatched for binding until it meets them:

1. **Names which quantity is fixed structure and which is trainable, in that
   order, before the trainable part is scored.**
2. **Grounded in fetched prior art with exact figures** — page, section, quoted
   line, never a paraphrase from memory.
3. **Reframes an existing object rather than patching one** — states what the
   thing provably is, not a special case kept alive to save an old story.
4. **Falsifiable by a measurement that already exists**, or names the one that
   would falsify it.
5. **The proposer ran their own instance and reported it when it failed**, in the
   record, before dispatch — not only the passing numbers.

Applied to the four leaps this project has produced:

| leap | score | outcome it had |
|---|---|---|
| Chebyshev degree law | **5/5** | exact — reproduced 11/34/130 with no fitted constant |
| segment bit | 4/5 | bound by RED test, then conceded as Blelloch 1989 |
| three-channel dtype gate | 4/5 | half alive — bf16 mantissa died at S = 75 |
| zero-density account | **2/5** | refuted by the table it was built to explain |

**The gate retrodicts the outcomes**, which is the only evidence that it is a gate
and not a preference. Condition 5 is the one the project has never enforced and
the one both exemplars satisfy: the S-matrix instance that diverged at
`ρ = 1.6688`, with `E` inside the band, was reported corrected and **unscored**
rather than quietly moved.

The band-position leap now awaiting its free kill scores **3/5 pending fetch** —
it names fixed against trainable explicitly, it reframes, and its step 0 is
falsifiable off an existing checkpoint, but six of its citations are from memory
and unfetched, and it borrows another instance rather than running its own.

---

## 14. The band-position leap, and a kill that could not fail

The leap: `γ·ρ(P) < 1` — what this project calls exactness — is in physics the
statement that **E is off the spectrum**, and a resolvent off the spectrum does
not propagate, it images. It offers one fixed fact under three findings:
representation is near-field, prediction is far-field transport, and a gate past
the decay length multiplies a contribution with zero expected signal.

**Its free kill was written so that it could only ever return one answer.**

The instruction was to read `γ·ρ(P)` off the trained checkpoint and compare
`ξ = −1/ln(γρ)` against the support-graph diameter. But **`P` is row-stochastic,
so `ρ(P) = 1.0` exactly by Perron–Frobenius** — for every `P` this codebase
builds, trained or untrained. That fixes `ξ(P) = −1/ln(0.9) = 9.4912`, constant,
always above a diameter measured at 1 to 3. The leap dies **without a single
trained weight being consulted**, which is not a measurement.

The dispatch also asserted a diameter of "about 7, for a king-move board". `n`
here is a chart-position count — 16 or 32 — not chessboard squares, and the
operator's support graph is far denser than king-move adjacency.

### On the correct object, the leap survives

`Q` is the sub-stochastic transient block that the model's own committor solve
`(I − Q)⁻¹ R` actually uses, and it is not pinned to 1:

| | primary checkpoint | cross-check |
|---|---|---|
| `ρ(Q)` | mean `0.0243`, min `0.00005`, **max `0.2315`** | `0.1944` |
| `ξ(Q)` worst case | **`0.637`** | **`0.574`** |
| diameter(`Q`) | 1 – 2.47 | 1 – 2.00 |

`ξ(Q) < diameter(Q)` at every threshold on both checkpoints, so **step 0 passes
on the object the model actually solves**, and the largest measured `ρ(Q)` of
`0.2315` sits close to the leap's own cited `0.2596`.

One caveat carried rather than buried: the cross-check checkpoint returns
`ρ(Q) = 0.1944` **identical across all 64 examples**, which is the signature of a
documented pre-2026-09-08 dead-per-example-operator defect, so it is treated as a
single degenerate data point rather than 64.

### Both free reads are blocked, and were reported blocked

The gate corollary's two reads — open-gate density by hop distance, and per-step
gate ratios — both presuppose a discrete hop-binned gate attached to the chess
checkpoint's `P`. **That `P` has no gates**; it is a continuous softmax. The only
real gate system in the repository trains on an unrelated synthetic corpus with
no chess and no per-step log. The data does not exist, and the reads were
returned as blocked rather than approximated.

### The citations, fetched

| source | bibliography | content claim |
|---|---|---|
| Anderson 1958, Phys. Rev. **109**, 1492 | confirmed, title exact | abstract confirms localization; the *locator-expansion* phrasing is **UNREACHED**, full text paywalled |
| Weinberg 1963, Phys. Rev. **131**, 440 | confirmed, "Quasiparticles and the Born Series" | **abstract confirms** the paraphrase — the series fails when bound states are present |
| Combes–Thomas 1973, Commun. Math. Phys. **34**, 251–270 | confirmed — **volume and pages were unstated in the leap and are supplied here for the first time** | UNREACHED at this pass — **later reached via Project Euclid and found MISATTRIBUTED, see §15** |
| Lewontin–Cohen 1969, PNAS **62**, 1056–1060 | confirmed, title exact | **UNREACHED**, 403 |
| Ash–Nicholls 1972, Nature **237**, 510–512 | confirmed, exact pages | **UNREACHED**, login wall |
| Synge 1928 | **UNREACHED** — no direct source found | — |

So condition 2 of the hop gate passes on bibliography and **fails on content**:
the two load-bearing claims, Combes–Thomas exponential decay off-spectrum and the
Lewontin–Cohen drift correction, remain unverified beyond their titles. Neither
enters a numeric argument until it is read.

> **§15 supersedes this row.** Both were re-fetched through open archives. One is
> misattributed outright; the other splits, with its qualitative claim verified
> from its own abstract and its quantity still unreached.

---

## 15. The open archives, and the sixth misattribution

The first fetch confirmed five citations bibliographically and stopped at
publisher paywalls on content. Mathematics of that age is open — PNAS 1969 is on
PubMed Central, Communications in Mathematical Physics 1973 is on Project Euclid
— so the rule gains a clause: **for an old paper the publisher is the last route,
not the first, and `UNREACHED` is only honest after the open archive has failed
too.**

Project Euclid opened. It produced a misattribution.

### Combes–Thomas 1973 does not state the claim attached to it

The full 20-page PDF was retrieved and searched. **`pdftotext -layout` plus a
whole-document grep for `resolvent`, `kernel` and `Green` returns zero hits for
all three.** The paper never states a bound of the form
`|G(x,y;E)| ≤ C·e^{−κ|x−y|}` for the resolvent kernel at energies off the
spectrum.

What it proves is **exponential decay of eigenfunctions**. Theorem 1, p. 257: if
two-body interactions are boost-analytic and `ψ` satisfies `Hψ = Eψ` with
`E < E₀ = inf σ_e(H)`, then `ψ ∈ D(e^{θ√(2M(E₀−E))R})` for any `0 ≤ θ < 1`.
Theorem 2 extends the rate to the distance from the nearest threshold. That is
decay of bound-state wavefunctions **at their own eigenvalues**, established
through dilation-analytic continuation and the meromorphy of `(H(γ) − z)⁻¹`.

**The resolvent-kernel bound is a later generalization that carries their name.**
The "Combes–Thomas estimate" is a real and standard result; this 1973 paper is
not where it is stated. The leap cited the name correctly and the paper
incorrectly — **the sixth misattribution in five days, by the same mechanism
every time: repeating what a thing is called instead of reading what it says.**

The load-bearing claim therefore has no verified primary source yet, and no
numeric argument may rest on it until the actual source of the resolvent bound is
fetched.

### Lewontin–Cohen splits

**Verified, primary** — the abstract, read independently on PubMed Central and on
PNAS: extinction probability approaches unity even as expected population size
grows without limit, *"owing to the difference between the geometric and
arithmetic mean growth rates."*

**UNREACHED** — the `σ²/2` drift correction. Pages 1057–1060 exist on both hosts
only as non-OCR scanned images, and every route to the PDF hit bot detection: a
proof-of-work and reCAPTCHA gate on PMC, a Cloudflare check on PNAS. **Those were
not bypassed**, correctly. Nine routes were tried and recorded, including a
Wayback snapshot that carries the same un-OCR'd images and a Europe PMC render
endpoint that returns 403 server-side and a 537-byte empty payload in a browser.

So the qualitative claim — a positive arithmetic mean is compatible with certain
extinction — **stands on its own paper's abstract**. The quantity does not, and
the leap's use of `σ²/2` is unsourced until those four pages are read.

### One route fact worth keeping

**Project Euclid's landing page states the article is available only to
subscribers, while the direct PDF URL on the same domain serves the full
document.** The paywall was in the page, not in the file. A lane that reads only
the landing page records `UNREACHED` for an open paper.

---

## 16. Q2: the first row entitled to a generalization sentence, and it did not tie

Every language-model comparison this project had made sat at 5.4× to 28.3× corpus
repetition — the memorization regime, where a gate that helps or hurts says
nothing about generalization. **This row sits below 1× and is the first that can
support the sentence.**

Fixed structure printed before any arm was scored, verified in the board log by
ordering: `q3_fixed_structure` at line 4296 precedes `start_variant` at 4300;
`q2_fixed_structure` at 4416 precedes the first `q2_seed_done` at 4417.

| | arm (a) softmax twin | arm (f) hard-concrete |
|---|---|---|
| params by `numel()` | **724,608** | **725,391** |
| repetition | **0.7977×** | **0.7985×** |
| attention | `F.scaled_dot_product_attention`, `is_causal=True` | `arm_smprime` gated readout |

The arms differ by **783 parameters, 0.108%** — and that excess is exactly
`3 × (2·(128+1) + 3)`, the per-block quantity `modeling_ceq.py:409-418` already
documents for `m_head`, `theta_head`, `beta`, `qk` and `g`. The softmax twin was
checked for dead weight: at `operator="sgate"` only `qkv` and `o_proj` are built
and `softmax_forward` uses both, so it is not a padded skeleton. **0.108% cannot
carry 0.247 nats.**

### The result

```
                seeds 0-4 final eval loss                     mean
softmax  (a)    1.2757  1.2775  1.3021  1.3013  1.2916       1.2896
hard-cc  (f)    1.0431  1.0424  1.0430  1.0392  1.0439       1.0423
diff            -0.2326 -0.2351 -0.2591 -0.2622 -0.2476      -0.2473
```

**The gated arm wins at 5 of 5 seeds, and the smallest gap is 23× the
pre-registered tie band of 0.01 nats.** Both the tie branch and the
gate-costs-generalization branch are excluded.

**This is not written as a win.** The pre-registration says a 5-of-5 result is
the first learned win this project would have **and is not written until verified
on a second corpus**. It is recorded here as a measurement. The verification is
the next row, not a formality.

### Q3: the gate sweep, at five seeds, with no mechanism attached

| form | last-50 loss | trained exact-zero | backward reach mean | live gradient |
|---|---|---|---|---|
| clamp | 1.2604 | 0.2760 | 32.33 | 0.5514 |
| straight-through | 1.2843 | 0.3912 | 1.32 | 1.0000 |
| **hard-concrete** | **1.1218** | **0.0018** | **214.35** | 0.9839 |

Hard-concrete takes the lowest loss **and** the highest reach at **every seed
individually**, so the seed-dependence counter does not fire. Seed 0 reproduces
the single-seed reference almost exactly.

**No mechanism sentence appears anywhere in this row.** Four accounts of gate
closure are already dead; a fifth was not born from a table.

### Two constants that did not survive being measured

**The certificate parameter count was wrong.** `632,496` was carried into this
page and into a commit message. Measured `numel()` is **724,608** — a **0.5%
excess** over the closed form's `720,896`, not a 12.3% shortfall. The conclusion
survives, since both arms remain far below 1× repetition, but the number was
quoted rather than audited, which is the exact defect L-REFLECTOR names.

**And the clamp reach constant reproduces at no seed.** Per-seed `reach_mean` for
the clamp arm reads `2.1, 26.5, 65.3, 1.8, 66.0`. The `2.146` quoted throughout
this phase is **one draw from a distribution spanning 1.8 to 66.0**, and every
sentence resting on clamp's reach being *small* rests on a single seed.
Hard-concrete's reach, by contrast, is stable: `241.4, 200.2, 206.1, 221.5,
202.5`.

---

## 17. The second corpus: it transfers, and the gap grows

The pre-registration held the Q2 result back from being called a win until a
second corpus confirmed it. A second **domain** now has.

| corpus | repetition | (f) − (a), 5 seeds | mean |
|---|---|---|---|
| TinyStories — synthetic children's prose | 0.7977× / 0.7985× | −0.2326 … −0.2622 | **−0.2473** |
| WikiText-103-raw — encyclopedic English | 0.2155× / 0.2158× | −0.2977 … −0.3508 | **−0.3285** |

**Five of five seeds on both, and the gap is 30× the pre-registered tie band of
0.01 nats. The effect is larger on the harder domain, not smaller.**

Both corpora sit below 1× repetition, so neither run is in the memorization
regime and a generalization sentence is admissible for both.

The corpus is pinned rather than named: WikiText-103-raw from
`Salesforce/wikitext`, CC BY-SA 4.0, `105,326,029` bytes on disk
(`sha256 3523ac1d…`), of which the **unchanged** 64 MiB cap in
`train_with_eval` loads `67,239,422` (`sha256 3b81cdca…`), split by document into
3,866 articles — 3,479 train, 387 validation.

**Exactly one variable moved**: the corpus file and its token count. The shape,
the initializer, the split convention, the seed set, the optimiser, the steps
formula and the byte-level vocabulary are the values `q2_certificate.py` locks,
and `r3_eval.py` and `r1_gate.py` were reused unedited.

### What this does not settle

`split_seed = 0` is fixed across all five seeds **on both corpora**. Every
interval here carries model variance and **no eval-draw variance at all**. So the
result now holds across two domains and remains untested against the draw of the
held-out set itself.

That is precisely the contrast `C_win` measures in the 4×5 paired design, where
the split seed varies with the model seed and the arms are paired by common
random numbers. Until it reports, the honest statement is: **the effect survives
a domain change at five of five seeds, twice, and has never been measured against
a second validation draw.**

---

## 18. The mass-gate arm is a replication, and the page says so first

Arm (a2) — an SDPA twin plus a per-head sigmoid output gate — was gated behind
three citations reaching the page **before** it ran. The reason is not ceremony:
if the 0.247 nats turns out to be normalization, then it is published work, and a
project that measures first and discovers that second has nothing to say.

All three are now fetched and quoted.

**Qwen Team et al., arXiv:2505.06708v1 (10 May 2025), NeurIPS 2025 Oral —
"Gated Attention for Large Language Models: Non-linearity, Sparsity, and
Attention-Sink-Free".** Abstract, verbatim: *"Our central finding is that a
simple modification — applying a head-specific sigmoid gate after the Scaled
Dot-Product Attention (SDPA) — consistently improves performance."*

Their §2.2 and Figure 1 place the gate at position **G1**: after the SDPA outputs
are concatenated across heads and **before the output projection `W_o`**. A
separate position G5 applies it *after* `W_o`, and their own ablation finds G1
the stronger of the two. **Before-versus-after `o_proj` is the axis their paper
turns on, and (a2) sits on their winning side.**

**So (a2) is a replication, close to an exact one**, and `C_mass` prices a known
mechanism at toy scale on a different bed. That sentence is on the page before
the arm runs rather than after it reports.

**Evan Miller, "Attention Is Off By One" (24 July 2023)** — an essay, not a
paper. `softmax₁(x)ᵢ = exp(xᵢ) / (1 + Σⱼ exp(xⱼ))`, and verbatim: *"All I did was
added one to the denominator. This lets the vector as a whole tend to zero if it
wants."* The source's own non-standard grammar is preserved, and was used as
evidence the line is quoted rather than paraphrased.

**Xiao et al., arXiv:2309.17453v4 — attention sinks.** Fetched directly after the
first attempt declined it.

**Donsker–Varadhan** — Dupuis and Ellis, *A Weak Convergence Approach to the
Theory of Large Deviations*, Springer 2011, **Lemma 1.4.3, p. 405**.

### A confound in the contrast, found before the night rather than after

**(a) and (a2) are not parameter-matched**: `724,608` against `727,704`, a
difference of **3,096 parameters, +0.43%**. Arms (f) and (f0) *are* matched at
`725,391` either side.

So `C_mass` as built prices **gate plus parameters**, not the gate alone. The
night runs with that stated beside the contrast rather than delayed to construct
a dead-gate parameter-matched (a): 0.43% of parameters against an effect of
roughly 0.25 nats is a weak confound, and naming it costs nothing while stalling
costs a night.

### A dispatch defect, recorded because it is recurring

One fetch was **declined outright** by a lane that read a mid-turn message
addressed to the orchestrator as its own instruction, and reported the source as
not attempted. It was overridden and fetched. This is the fifth instance this
session of a relayed message reaching a lane's prompt and displacing its task.
The defect is in the dispatch template, not in the lane.

---

## 19. Step 0: row mass is not one, and closure is not the gate

The question was whether arm (f)'s advantage could be an un-normalized degree of
freedom. `W_ij = G_ij·e^{s_ij} / Z_i^β` with `Z_i = Σ_j |G_ij| e^{s_ij}` — the
numerator complex, the denominator summing the modulus — confirmed in source at
`arm_smprime.py:302, :324, :345`. At β = 1 a row sums to one only when every
phase in it is equal.

**It does not sum to one.**

| layer | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| mean row mass | 0.9607 | **0.7930** | **0.7971** | **0.7423** |
| deficit | 0.039 | 0.207 | 0.203 | 0.258 |
| rows below half mass | — | **13.2%** | **11.3%** | **16.5%** |

Minimum row mass **0.0023**. At layer 3, **190 rows sit in [0.0, 0.1)** and only
**2,510 of 32,768** rows exceed 0.99. This is not a rounding tail.

A genuine `scaled_dot_product_attention` row sums to exactly one **by
construction**, so arm (f) owns a learned, per-position, un-normalized channel
the twin cannot represent at any setting.

### β is a second mass channel, and it was never priced

```
beta   L0 1.090703   L1 0.961642   L2 0.962446   L3 0.991261
```

The direction is consistent across **all five seeds**: L0 rises in every one
(1.0683–1.1062), L1–L3 fall in every one (0.946–0.983). `beta_census` runs
10.5–24.5 per layer per seed, so the dial was exercised rather than left pinned
for want of signal. `qk` moved to 0.894–1.064 and `g` to 0.941–0.973.

Since `Z_i` is a data-dependent positive row sum, **β ≠ 1 rescales every row by
`Z_i^{1−β}`** on top of the phase-cancellation deficit — a second multiplicative
per-row mass channel, also outside softmax's reach.

### Closure is underflow; mass is phase

The exact-zero fraction of the gate `m` itself reads `0.0, 0.0024, 0.0, 0.0` by
layer — **the gate barely closes**. Yet **59–75% of causal `W` entries are
exactly zero**, and at layer 0 that is **100% float32 underflow of the path
product**, since `m` has no exact zeros there at all.

**These are two different mechanisms and the project has been treating them as
one.** The refusal channel this operator is named for is, in a trained model,
floating-point underflow of a product — the same defect §2 measured in isolation,
now measured in place.

And **56.7 / 72.8 / 78.5 / 79.2 percent of row-mass variance sits on the head
axis** while `m_head` and `theta_head` are `nn.Linear(d, 1)` and **head-blind**.
The per-head `exp(qk·q·k)` factor is what splits it, so the mass channel is not
purely a gate property either.

### The prediction, logged before the night runs

`C_phase` and `C_mass` are predicted **non-zero**. Row mass departs from one by
0.04 to 0.26 by layer with a minimum near zero, and β has moved off one in a
consistent direction at every seed, so both the phase channel and an output
mass-gate have something real to price. Had row mass come back pinned at 1.000
this would have been dead for free; it did not.

### Two defects found on the way

**`train_with_eval` never saves a checkpoint.** `torch.save` does not appear in
it — grep-confirmed — so all five arm (f) models from the certificate row were
discarded in-process, and the five `q2_ckpt_hardconcrete_seed*` directories hold
only `run_record.json`. The numbers above are therefore read from the **gate-sweep**
checkpoint, not from the arm that produced the 0.247, and that substitution is
stated rather than glossed. Any future row wanting arm (f)'s own weights must
re-train.

**The row stopped rather than rebuilding.** Told no checkpoint existed, the lane
reported it and halted with a non-zero exit instead of reconstructing a model and
reading numbers off it. A rebuilt model would have produced plausible figures for
an arm that no longer exists.

---

## 20. The floors: the chess numbers were computed at the wrong n

Two floors from the same generation, and this project had been scoring against
neither.

**Wiener 1942 reproduces.** `0.4603 / 0.3480 / 0.2506` against the specified
`0.4594 / 0.3498 / 0.2502` — within 0.5% on every term, one run, no tuning
(`default_rng(0)`, AR(2) `a = (1.2, −0.5)`, process sd 0.5, observation sd 0.3,
N = 200,000). At p = 32 the answer is identical to p = 8 to four decimals, so an
order-8 linear predictor already captures this process fully and the gap to
oracle stays 0.0974 at both orders.

**Cramér–Rao 1945 reproduces exactly.**

| p | ε | n (CR/CLT) | n (Hoeffding) | ratio |
|---|---|---|---|---|
| 0.50 | 0.02 | 2,401 | 4,612 | 1.92× |
| 0.20 | 0.02 | — | — | 3.00× |
| 0.10 | 0.02 | 865 | 4,612 | 5.33× |
| 0.05 | 0.02 | — | — | 10.11× |
| **0.01** | 0.02 | — | — | **48.50×** |

The loosest corner is the rare-event one. Hoeffding is tightest at p = 0.5, its
own calibration point.

### But L-WIENER attaches to no bed in this repository

The chess bed has **no cross-game time index**: `chess.py:117` gives *every ply of
one game the same one-hot outcome label*, and `ply_idx` resets per game carrying
no meaning across games. A linear-predictor floor needs a sequence, and there
isn't one spanning the rows.

**And MDP-CAL does not exist.** An exhaustive search of `ceq/` and `ceqjepa/`,
source and git history, finds no file, class, function or variable by any
spelling of it. BED-M, BED-H, `LSTD_bed` and a gridworld exist; none is an MDP
calibration bed. It has been named in contracts as though it were one.

### The n was positions, and positions are not independent

**Every published chess resolution number used the position count as `n`.** But
all plies of a game share one outcome label, so positions within a game are
perfectly correlated. `n = 6000 positions` is roughly 1,000 independent draws,
and correcting `n` from positions to games **flips `oracle_ceiling_RES` from 13.7×
above its Cramér–Rao floor to below it.**

Three rows sit below their own floor at the n actually used:

| row | value | n used | against naive p=0.5 floor 0.01265 |
|---|---|---|---|
| `operator_committor.RES` | 0.0012548892 | 6,000 positions | **~10× below** |
| `draw.res_k` | 0.0004429054 | 6,000 | below |
| `sink.res_k` | 0.0007011879 | 6,000 | below |

A row below its floor is reporting noise with a confidence interval drawn around
it.

### And the numbers this page carried are not the numbers in the file

| quoted throughout this phase | actually in `wil_chess400_results.json` |
|---|---|
| RES `0.001469` | **`0.0012548892`** |
| oracle ceiling `0.10117` | **`0.0241166938`** |

Worse, three result files cited as sources **were never produced**:
`wil_res_ceiling.json`, `wil_bar_can_fire.json` and `wil_recal_race.json` do not
exist. Those figures came from lane reports rather than from disk, and were
published as measurements.

**What survives and what does not.** The prediction verdict survives — the
operator's committor was tied by an eight-bin histogram of its own input, and a
tie is a tie whatever the ceiling. **Its quantification does not.** "1.45% of the
bed's resolution ceiling" is meaningless when the ceiling is itself below its
noise floor at the corrected n, and the two figures that sentence was built from
do not match their own file.

Every resolution sentence in `PHASE_I.md` §1 and `PHASE_I1.md` §8 is struck to a
direction without a magnitude until re-measured at `n = games`.

---

## 21. Rows still out

`R1` gate parameterization (straight-through against hard-concrete, with the
four-condition must-fire); `R3` held-out eval path by document and `R4`
determinism; `R5` hardware assertions and `R6` dtype guards, each of which must
be demonstrated firing rather than described.

**Kaggle remains closed** on the verified absence of any held-out quantity, until
`R3` lands.
