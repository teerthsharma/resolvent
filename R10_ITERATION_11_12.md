# R10 — iterations 11 & 12: two verdicts, and two rules that do not hold up

Both iterations reached the verdict their script asked for. Both also produced a
finding about the rule they were judged by, which is the more durable half.

---

## it.11 — the region is learnable

**The rule, pre-registered.** *"ALL: verdict — region learnable iff ≥1 cell < 1.0
with N=8 seed CI excluding 1.0 (bootstrap, B=10⁴); adjudicated against Venus's
filing; branch recorded."*

**The verdict.** MET, at `(t*=2, n=2048, steps=150)`:

| quantity | value |
|---|---|
| mean over seeds | 0.9523 |
| seed CI, B=10,000 | **[0.9462, 0.9591]** |
| N | 8 distinct seeds, at fixed threads=6 |
| excludes 1.0 | **yes** |

Companion readings at the same rung: `t*=8` [1.1162, 1.1314] and `t*=32`
[1.1307, 1.1837], both **above** the bar. That contrast is the whole reason the
n-axis mattered: at n=2048 only `t*=2` clears, and the larger rungs are what move
`t*=8`.

**Why `t*=2` counts.** Softmax's hop budget is **1** `scale/e_ladder.py:55` (`HOP_BUDGET = {"softmax": 1, ...}`, consumed at `:202`), so
`t* > hop` is satisfied by all three blocks. Venus's filing assumed hop=2 and
therefore treated `t*=2` as outside the region; it is inside it.

**Scope at the time of writing, and how it closed.** This verdict did **not**
establish the `t*=8` crossing at n=32768 — a finer claim, then resting on 4 seeds,
which `scale/it11_verdict.py` refused to rule on rather than report a CI over an N
nobody registered.

**The seeds were subsequently bought and that cell now clears too:**

| cell | N | seed CI | it.11 |
|---|---|---|---|
| `t*=2`, n=2048 | 8 | [0.9462, 0.9591] | **LEARNABLE** |
| `t*=8`, n=32768 | 8 | **[0.9739, 0.9768]** | **LEARNABLE** |
| `t*=32`, n=32768 | 4/8 at the time | — | refused |

**That last row has since resolved.** The seeds were bought and `t*=32` at n=32768
reads **NOT LEARNABLE**, seed CI **[1.0009, 1.0057]** over N=8 — a verdict, not a
refusal, because the interval lies entirely *above* the bar. The row is left as
written because it records what the gate did at the time, and the refusal is the
behaviour worth preserving: it declined at 4 seeds and the answer it eventually
gave was not the one it would have given early.

So the open region is established as learnable at **two independent points**, at
opposite ends of the data axis, both fully compliant. The refusal machinery did
its job in between: it never reported an interval on 4 seeds, and the verdict it
eventually gave was not the one it would have given early.

**The instrument.** `scale/it11_verdict.py` computes the interval over the
population the rule names — seeds — rather than the one the journals ship. It
refuses below N=8, counts distinct seeds rather than rows, and builds the
interval at a fixed thread count. Its `demo()` fails if any of those three
protections is removed.

### Correction, against this record's author

`verdict()` filtered on `(t*, n)` and **not on `steps`**, pooling the 150- and
600-step rungs so two different cells sharing a seed read as one seed recorded
twice with different values. It was accidentally correct at n=32768, where only
steps=150 ever ran, and wrong at every rung with more than one. It was caught by
the module's own `by_seed` guard firing on its caller.

`demo()` then passed synthetic rows with no `threads` field and broke the moment
`by_seed` began grouping on it — a self-check that did not exercise the shape its
own module reads. Both repaired, with a must-fire added for the missing-field
case.

---

## it.12 — the E4′ strike stands; its stated confidence does not

**The rule.** *"SATURN: E4′ spread — 20 draw seeds, report mean μ̂, sd σ̂, and
z = (shipped − μ̂)/σ̂ (currently −0.78 on one draw); strike stands iff sign(effect)
holds in ≥ 18/20 (binomial p = C(20,≤2)·0.5²⁰ ≈ 2.0e-4 under fair coin), else
withdrawn."*

**The measurement**, block A, `case.seed` `0x33960007..0x3396001A`:

| quantity | value |
|---|---|
| μ̂ | 0.479472 |
| σ̂ (ddof=1) | 0.011083 |
| z = (0.471045 − μ̂)/σ̂ | **−0.7603** |
| k | **19 / 20**, errors 0/20 |

**Verdict, applied mechanically: 19 ≥ 18, STRIKE STANDS.**

Harness `scale/e4prime_spread.py`, byte-identical on re-run, reproducing the
shipped 0.4710453160486452 bit-for-bit. The draw seed is reachable without
editing tracked source — `draw_balanced_marginal` reads `case.seed` as a plain
field — and the script asserts `edges`/`partition` identity per row so only the
draw moves.

### Block A was a reproduction, not an independent twenty

SATURN reported this without being asked. Four of MARS's statistics matched to
six decimals — min 0.459390, max 0.506250, mean 0.479472, and the reversal count
— which means the declared seed rule *reselected his seeds*. A block that agrees
to six decimals is not confirmation; it is the same measurement.

So a disjoint block B was run (`0x33960100..0x33960113`):

| block | μ̂ | σ̂ | reversals |
|---|---|---|---|
| A (= MARS's seeds) | 0.479472 | 0.011083 | 1 / 20 |
| B (disjoint) | 0.475633 | 0.010301 | **0 / 20** |
| pooled 40 | 0.477553 | 0.010738 | 1 / 40, bar at 2.09 sd |

**A correction to MARS.** He reported the **population** sd 0.010802 where the
sample sd is 0.011083 (`0.011083 × sqrt(19/20) = 0.010802`), making the spread
read 2.6% tighter than it is.

### The rule cites a null its own design cannot support

`C(20,≤2)·0.5²⁰ = 2.012e-4` is arithmetically correct and is the p-value of a
**sign test**. Nothing makes 0.5 the null crossing rate for a continuous NRMSE
against a fixed pre-registered bar, and the 20 draws share one graph, one
partition and one split, so they are not 20 independent replications of the
effect.

The decision-relevant quantity is `P(score >= 0.5)`:

| sample | estimate | Clopper–Pearson 95% |
|---|---|---|
| 20 draws | 1/20 = 0.05 | [0.0013, 0.2487] |
| 40 draws | 1/40 = 0.025 | [0.0006, 0.1316] |

**The rule cites a p three orders of magnitude tighter than the design supports.**
The strike still stands on the `k >= 18` threshold, which is a decision rule and
needs no null. What does not stand is the confidence the script attaches to it.

### The instrument was shown able to fail

Required before trusting a sweep in which almost nothing crosses. The harness
returns ≥ 0.5 twice: seed `0x33960007` inside the sweep at 0.506250, and
`StableSparse_S2Rips_64` at 0.892650, matching the value asserted at
`tests/cameron/test_e4_rips_gate.py:147`. Without that, "no seed crosses" and "the
rig cannot cross" would be the same reading.

---

## What both iterations have in common

Each was scored by a pre-registered rule, and in each case the rule survived as a
**decision procedure** while failing as a **statistical claim**:

- it.11's threshold (N=8, CI excluding 1.0) is met, and the shipped
  `boot_lo`/`boot_hi` that a reader would naturally check is an interval over the
  wrong population (instance 16).
- it.12's threshold (k ≥ 18) is met, and the p-value the rule quotes for it is
  wrong by ~10³.

Pre-registration bought the thing it is supposed to buy — nobody chose the
threshold after seeing the data. It did not make the threshold's justification
correct, and this round is now two-for-two on rules whose arithmetic was never
checked against their own design.
