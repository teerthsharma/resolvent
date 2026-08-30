# MERCURY (LESTRADE) — R9 iteration 3

Role: runner. Prices, schedules, executes. **Nothing below is a verdict.** The
pre-registered rows A–E were filed by Mars before any datum existed; section 3
prints what each row's condition reads and stops there.

**HEAD check.** Session opened at `7739a24`, fast-forwarded to
`feat/r9-causal-consequence` @ `7af6dc0` before any file was touched.

Box unchanged: Windows 11, Raptor Lake, RTX 4060 Laptop, torch 2.5.1+cu121,
`torch.set_num_threads(2)` pinned in-module.

---

## 1. Pre-flight, before the first number

### 1.1 The gating bind holds

Mars reported `_bind_batched_against_arm_s` returning `False` on its
**diagnostic** shapes and asked that the gating shapes be re-verified. Re-run
this session:

```
map bind n=2 s=64 d=24 k=8 seed=0: BITWISE
map bind n=2 s=64 d=24 k=8 seed=1: BITWISE
  gating bind bitwise: True
```

The diagnostic failure reproduces exactly as Mars described — `n=2, s=128,
d=16, k=32, seed=2`, max abs diff `1.387779e-17`, and two further sub-threshold
`av` differences at `n=3, s=64, d=24, k=16, seed=1` of `8.673617e-18` and
`1.387779e-17`. Pre-existing, non-gating, **untouched** per hard rule 2.

### 1.2 Ceiling arithmetic, printed before any `argmaxste` datum

`LOOP_PROMPT 1.3`. Computed from the journal alone; no `argmaxste` value
existed when this was produced.

| quantity | value | source |
|---|---|---|
| pre-registered 13-seed resolution | `0.027260` | `scale/e_ladder.py:96` |
| prose floor at 5 seeds | "roughly 0.05 NRMSE" | `M3_QUINTUPLE_PREREGISTERED_READING.md:88-90` |
| `argmax` mean, 5 seeds | `1.010779` | journal |
| `softmax` mean, 5 seeds | `0.892323` | journal |
| the disputed gap | `+0.118456` | `1.010779 − 0.892323` |
| **measured floor for this contrast** | **`0.015956`** | paired bootstrap CI half-width on the journalled `argmax`/`softmax` pair, B=10000, seed 0 |

The measured floor is **finer** than `RESOLUTION_13`, not coarser: `0.015956`
against `0.027260`, a factor of `1.71`. The prose floor of "roughly 0.05" is a
worst-case statement across contrasts; this particular pair has a low paired
spread, so the achievable resolution here is better than the generic figure and
better than the 13-seed target. **That is the opposite of what the dispatch
anticipated, and it is stated here rather than discovered afterwards.**

What that buys, row by row, computed before the data:

- **Row A** is resolvable only *as a bound*. A CI covering zero cannot separate
  "training the selection buys nothing" from "training buys up to `0.015956`",
  i.e. up to **13.5 %** of the disputed gap. Row A must never be reported
  without that bound attached.
- **Row B** needs both halves resolved, so the split must land in
  `[0.015956, 0.102500]` — a window `0.086544` wide, **73.1 %** of the gap.
  Comfortably scoreable.
- **Row C** needs a move of at least `0.102500` toward softmax.
- **Row D** needs `0.015956` the wrong way.
- **Row E** is a threshold read, not a contrast, and is always decidable.
  `argmax` reads `1.010779`, so row E is live exactly as Mars said.

A null on any row is evidence of "no effect larger than `0.015956`", never of
no effect.

---

## 2. The STE reading — RUN, 5 seeds, complete

`negation_scope` at `s64_d24_st150_ntr8192_nev512_b21`, the geometry the disputed
headline was measured on. `argmax`, `softmax` and `twin` were already journalled
there; only `argmaxste` was run, so every contrast is paired on identical seeds.

| cell | mean | seed 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| `softmax` | `0.892323` | 0.877168 | 0.889523 | 0.919148 | 0.890175 | 0.885603 |
| `twin` | `0.780927` | 0.767403 | 0.784397 | 0.794505 | 0.798001 | 0.760328 |
| `argmax` | `1.010779` | 1.022910 | 0.994867 | 1.011705 | 1.010804 | 1.013609 |
| **`argmaxste`** | **`0.785019`** | 0.801954 | 0.777060 | 0.746923 | 0.802827 | 0.796330 |

Contrasts, paired percentile bootstrap `B=10000`, seed 0, `delta = ref - arm`,
so a positive delta means the named arm has the lower error:

| arm | vs | delta | ci_lo | ci_hi | n+ | reads |
|---|---|---|---|---|---|---|
| `argmaxste` | `argmax` | `+0.225760` | `+0.212433` | `+0.245886` | **5/5** | excludes zero, arm ahead |
| `argmaxste` | `softmax` | `+0.107304` | `+0.082879` | `+0.140870` | **5/5** | excludes zero, arm ahead |
| `argmaxste` | `twin` | `-0.004092` | `-0.029187` | `+0.023107` | 2/5 | NO DIFFERENCE |
| `argmax` | `softmax` | `-0.118456` | `-0.134115` | `-0.102204` | 0/5 | excludes zero, ref ahead |

The last row reproduces the disputed headline exactly, on the same seeds, as a
control on the reading itself.

**A naming caution.** `verdict_of` labels its two directions `SETTLED WINS` and
`TWIN WINS` regardless of which cells are in the contrast, because it was
written for the settled/twin pair. Those strings mean only "the CI excludes zero
in the arm's favour" and "in the reference's favour"; they name no cell. The
column is relabelled here to say what it actually asserts, and the raw strings
are not quoted anywhere in this report.

### 2.1 Rows A-E, scored verbatim from `tests/mars/MARS_REPORT_IT2.md` section 6

| row | condition | reads |
|---|---|---|
| A | `argmaxste - argmax` CI covers zero | **False** |
| B | beats `argmax`, still above `softmax` mean | **False** |
| **C** | **reaches or beats `softmax`** | **TRUE** |
| D | worse than `argmax` | **False** |
| E | `argmaxste` mean at or above `1.0` | **False** - reads `0.785019` |

**Row E does not fire, so the contrast is creditable.** `argmaxste` clears
predict-the-mean by `0.214981`, where `argmax` fails it by `0.010779`. Row E was
the exhaustiveness catch and it was live; it is closed by measurement, not by
assumption.

**Row C is what fired.** Its licence, quoted from the pre-registration:

> The deficit was untrained selection. *"Reading one pivot is worse than reading
> none"* is **refuted**, and the mixture claim falls.

### 2.2 The size of it

The disputed gap was `argmax - softmax = +0.118456`. Training the selection did
not merely close it:

```
argmax    - argmaxste = +0.225760    190.6 % of the gap
argmaxste - softmax   = -0.107304    argmaxste is BELOW softmax
argmaxste - twin      = -0.004092    CI [-0.029187, +0.023107], covers zero
```

A one-hot lookup whose selection is trained lands **statistically
indistinguishable from the full mixture** - bounded by this run's own CI
half-width of `0.026147`, so the honest statement is "no difference larger than
`0.026147`", never "identical".

### 2.3 What this answers, and what it does not

The question was: *is the one-hot control worse than softmax because a lookup is
worse than a mixture, or because its selection was never trained?* The two cells
differ **only** in whether selection is trained - Mars bound the forward bitwise
to `argmax` (`torch.equal` 3/3, step-0 loss identical at `1.108632`) and
`n_params == 4769` holds on both.

The answer this run gives is **the selection**, at `n+ 5/5` on both halves. The
README's *"the gain is the mixture, not the equilibrium"* does not survive it:
the mixture is not what the `-0.118456` was measuring.

It says nothing about the equilibrium half of that sentence - `settled` was not
in this reading. It licenses dropping the mixture claim, not asserting its
converse, and the C1 lane in section 5 is what addresses the other half.

### 2.4 Clock, same-session only

The five units read `238.05 / 206 / 221 / 224 / 192` s, median `221 s`. The
journalled `twin` at this geometry reads median `508.74 s`, min `366.58 s`, but
those rows were taken in another session and **the coordinator's rule forbids
quoting that ratio**. No same-session `twin` unit exists, so whether `argmaxste`
is genuinely cheaper than `twin` or the box is simply fast right now is
**unresolved and not claimed here**. Mars priced `argmaxste` as `twin` in FLOPs,
which is an accounting claim and is untouched by this.

---

## 4. Venus's amendment, applied to the shared function

The coordinator's second change — print `n+` beside every CI — is a property of
every 5-seed reading in this repository, not of the C1 lane, so it went into
`contrast()` (`scale/m3_synthetic_settled.py:182`) rather than into the caller
about to be run. `per_seed_delta` was already returned; `n_pos` counts the seeds
on the winning side of it, and `scale/m3_quintuple.py`'s contrasts table prints
it as an `n+` column.

The mechanism was re-derived here in both directions rather than taken on
report: a 3–2 split produces `ci_lo < 0 < ci_hi` and `NO DIFFERENCE`, and a 5–0
split produces `ci_lo > 0`. Venus's 385/385, 20–44 % and 0–3.7 % figures were
**not** re-measured. `2 / 2**5 = 0.0625 > 0.05` is arithmetic.

**This applies to the STE reading in section 3 as much as to C1**, and every
interval there carries its count.

---

## 5. C1 pre-flight — verified, not started

### 5.1 The `_A` adapter's slice agrees with the dial, and the guard is not vacuous

`scale/m3_quintuple.py:656-666` derives `offset = s - yt.shape[1]` and
cross-checks it against `NS.e_t_star(task, s)`, raising on disagreement. All
four rungs agree exactly at `s = 64`:

| task | dial | label width | offset | agrees |
|---|---|---|---|---|
| `c1_propagate_t1` | 1 | 63 | 1 | True |
| `c1_propagate_t2` | 2 | 62 | 2 | True |
| `c1_propagate_t8` | 8 | 56 | 8 | True |
| `c1_propagate_t32` | 32 | 32 | 32 | True |

All four carry `E_T_STAR` entries, so the `dial is None` branch is not reached.
That mattered enough to check: had the C1 rungs been registered in `M3_TASKS`
without an `E_T_STAR` entry, the guard would have raised on **every** C1 run,
not only a misaligned one.

**Adversarial pass.** The guard is shown to fire, not assumed to:

```
mismatched dial   -> raises: label width 62 implies offset 2 at s=64,
                     but task 'c1_propagate_t2' declares t*=7; refusing to train ...
absent dial       -> raises: ... declares t*=None; refusing to train ...
```

### 5.2 The price, computed independently

Neptune's measured per-step costs at `s=64, d=24, n_train=2048`:

| cell | s/step | ×150 steps | 20 units (4 rungs × 5 seeds) |
|---|---|---|---|
| `softmax` | `0.006437` | `1.0 s` | `0.01 h` |
| `twinrow` | `0.666200` | `99.9 s` | `0.56 h` |
| `settledrow` | `4.560600` | `684.1 s` | `3.80 h` |
| **per rung-seed** | | **`785.0 s`** | **`4.36 h`** |

This reproduces the coordinator's `≈4.4 h` to within rounding.

**Adding `softmax` costs `19.3 s`, which is `0.123 %` of the table.** Venus's
amendment is free by any measure, and it buys `twinrow − softmax`, the only
contrast that can address the Bayes-optimality claim the lane exists to test.

**The 1.65× is already inside these numbers and must not be applied twice.**
`results/r9_perrow_pilot.md:137` states the per-step costs were measured *"on a
box currently running 1.65× slow"*. So `4.36 h` is the this-box price, and a
quiet box would read `2.64 h`. An earlier draft of this section was about to
raise a false alarm that the round should budget `7.2 h`; it should not.

The ruling's own arithmetic checks: 62 seeds on `t*=1` at `~784 s` per seed-rung
is `13.5 h`, against `4.4 h` for the whole 4-rung 5-seed table — **`3.10×` the
entire table for one rung.** Not running it is correct; dropping the rung would
be rung-picking, and labelling it UNDERPOWERED with the 62-seed requirement
stated is the third option the ruling takes.

### 5.3 A cost ratio that does not carry, again

`settledrow / twinrow` is `6.85×` in the vector lane. `settled / twin` at
`e3_t1` in the scalar lane is `2.11×`. The ratio changes by **`3.25×`** between
lanes. This is the third instance of iteration 2's M45 — an arm-cost ratio
measured on one lane cannot be carried to another — and it is the same shape as
the mistake type already logged for Saturn: pricing an arm at another arm's
rate.

### 5.4 Dispatch verified

`ALL_CELLS` carries all ten names including `argmaxste`, `twinrow` and
`settledrow`; all four `c1_propagate_t*` rungs are in `M3_TASKS` and reachable
through `--task`. The four commands are one per rung:

```
python scale/m3_quintuple.py --cells softmax twinrow settledrow \
  --seeds 0 1 2 3 4 --ks 8 --n-train 2048 --n-eval 2048 \
  --task c1_propagate_t<RUNG> --budget 100000
```

**IN FLIGHT, NOT COMPLETE.** Started the moment the STE reading finished, chained
so no box time was lost. Rung `t*=32` was running when this report was filed;
`t*=8`, `t*=2` and `t*=1` are queued behind it in that order, per the ruling to
prioritise the rungs where the theory predicts and the resolution is reachable.
**No C1 number appears in this report** -- a partial table is not a reading, and
the scorer prints each rung ceiling before that rung numbers when they land.
Starting C1 alongside the STE run
would have destroyed the only same-session clock either run has.

---

## 6. Claim ledger

| # | claim | class | check |
|---|---|---|---|
| M46 | The gating bind is BITWISE 2/2 at the run's own shapes | RUN | `s=64, d=24, k=8`, seeds 0 and 1, re-run this session before the first unit |
| M47 | Mars's diagnostic-shape failure reproduces and is pre-existing | RUN | `n=2, s=128, d=16, k=32, seed=2`, max abs diff `1.387779e-17`, plus `8.673617e-18` and `1.387779e-17` at `n=3, k=16, seed=1`. Non-gating; untouched |
| M48 | **The measured floor for the STE contrast is `0.015956`, FINER than `RESOLUTION_13`** | RUN | Paired bootstrap CI half-width on the journalled `argmax`/`softmax` pair, B=10000 seed 0. `0.015956` against `0.027260` is `1.71x` finer. Computed before any `argmaxste` datum |
| M49 | `argmaxste` reads `0.785019` over 5 seeds | RUN | Journalled: 0.801954 / 0.777060 / 0.746923 / 0.802827 / 0.796330 |
| M50 | **Row C fires; rows A, B, D, E do not** | RUN | Scored verbatim against `tests/mars/MARS_REPORT_IT2.md` section 6 |
| M51 | `argmaxste` beats `softmax` at `n+ 5/5` | RUN | `+0.107304`, CI `[+0.082879, +0.140870]` |
| M52 | `argmaxste` beats `argmax` at `n+ 5/5` | RUN | `+0.225760`, CI `[+0.212433, +0.245886]` |
| M53 | `argmaxste` is indistinguishable from `twin`, bounded at `0.026147` | RUN | `-0.004092`, CI `[-0.029187, +0.023107]`, `n+ 2/5`. The bound is the CI half-width |
| M54 | Row E does not void the contrast | RUN | `argmaxste` `0.785019 < 1.0`, clearing predict-the-mean by `0.214981` |
| M55 | The disputed headline reproduces exactly as a control | RUN | `argmax - softmax` `-0.118456`, CI `[-0.134115, -0.102204]`, `n+ 0/5` |
| M56 | All four C1 `_A` offsets agree with `e_t_star` | RUN | `1/63, 2/62, 8/56, 32/32` at `s=64`; all four rungs carry `E_T_STAR` entries |
| M57 | The `_A` guard is not vacuous | RUN | Raises on a mismatched dial (`t*=7`) and on an absent one (`t*=None`) |
| M58 | The C1 table prices at `4.36 h`, and the `1.65x` is already inside it | RUN + READ | Independent recomputation reproduces the coordinator's `~4.4 h`; `results/r9_perrow_pilot.md:137` states the per-step costs were measured on the slow box |
| M59 | Adding `softmax` costs `0.123 %` of the C1 table | DERIVED from READ | `0.006437 s/step x 150 x 20 units = 19.3 s` against `15,699 s` |
| M60 | The `62` seeds for `t*=1` reproduces from first principles | DERIVED | `(1.96 x 0.109 / 0.027260)^2 = 61.4 -> 62`, the standard paired-CI half-width count |
| M61 | `settledrow/twinrow` is `6.85x` against the scalar lane's `2.11x` | DERIVED from READ | A `3.25x` change between lanes; fourth instance of the arm-cost-ratio type |

**Adversarial pass.** The STE reading's own control is row 4 of the contrast
table: the disputed `-0.118456` reproduces to six decimals on the same seeds, so
the pipeline that produced the new number is the one that produced the old. Row
E was live and had to be checked, not assumed - had `argmaxste` landed above
`1.0` the whole contrast would have been void in either direction. The `_A`
guard was made to fire twice rather than observed to pass once. And the ceiling
was computed from the journal alone, before the first `argmaxste` unit existed,
which is why it could correct the dispatch's expectation rather than rationalise
the result.

---

## 7. What could not be validated

The STE reading is `negation_scope` at one geometry and five seeds; it contains
no `settled` cell, so it licenses dropping the mixture half of *"the gain is the
mixture, not the equilibrium"* and says nothing about the equilibrium half. The
`argmaxste - twin` null is bounded at `0.026147` by this run's own CI
half-width, so "indistinguishable" means "no difference larger than that" and
must never be read as "identical". Whether a trained one-hot selection also
matches the mixture at other tasks, other geometries or other `k` is untested -
one rung of one corpus is what was measured. The five unit clocks are
same-session; the journalled `twin` median at the same geometry is not, so no
cost ratio between `argmaxste` and `twin` is claimed, and Mars's FLOP-level
pricing of `argmaxste` as `twin` is an accounting claim this run neither
confirms nor disturbs. The `n+` column rests on Venus's 385/385, 20-44 % and
0-3.7 % measurement, which was not re-measured here; only the mechanism was
re-derived, and only at the two extreme regimes. The C1 lane is **in flight and
incomplete at the time of writing** - `t*=32` was running when this was filed,
`t*=8`, `t*=2` and `t*=1` are queued behind it, and no C1 number appears in this
report. Any C1 table that lands must be declared partial until all four rungs
are in; `verdict()` refuses quantified rows on incomplete data, so a partial
table cannot be misread as a kill, but a partial table is also not a reading.
The C1 price of `4.36 h` inherits Neptune's per-step figures unmeasured by this
session, and the `62`-seed requirement for `t*=1` inherits a realised
`sd_paired` upper bound of `0.109` that this session has not itself observed.
