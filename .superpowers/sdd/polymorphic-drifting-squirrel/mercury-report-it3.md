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
indistinguishable from the full mixture**. Stated in the house two-endpoint
form, because a symmetric half-width understates the wider side of an asymmetric
interval: the reading **excludes a `twin` advantage beyond `0.029187` and an
`argmaxste` advantage beyond `0.023107`, and excludes nothing smaller.** An
earlier draft of this report quoted a single half-width of `0.026147`; that
number understates the `twin` side, under which a true delta of `-0.029` is NOT
excluded, and it is withdrawn.

The interval itself sits on a **126-atom lattice**. Its exact, seed-free form is
`[-0.029187, +0.023107]` at B=10000 seed 0; see section 2.5 for why that
matters here.

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

### 2.5 The two published `argmaxste - argmax` pairs are one estimator

The exact pair on record is `[+0.212539, +0.245992]`; this session's
Monte-Carlo pair is `[+0.212433, +0.245886]`. The gap is `+1.056e-04` at **both**
endpoints, which is the signature of a constant estimator-family shift.

**It is not one.** Enumerating all `5**5 = 3125` resamples reproduces
`[+0.212539, +0.245992]` exactly, from an implementation written independently
of the one that produced it. Against that lattice:

| | Monte-Carlo, seed 0 | exact | on the lattice? | atoms apart |
|---|---|---|---|---|
| `ci_lo` | `+0.212433` | `+0.212539` | yes | 1 |
| `ci_hi` | `+0.245886` | `+0.245992` | yes | 1 |

Both Monte-Carlo endpoints sit **exactly on** the 126-atom lattice, each exactly
**one atom** below its exact counterpart. Sweeping bootstrap seeds 0..99 settles
it: `ci_lo` takes **2** distinct values and `ci_hi` takes **3**, reaching
`+0.2465162` — so **the endpoints move independently**. A constant shift cannot
do that. The equal gaps are equal local atom spacing at the two ends and nothing
more.

So for this cell the mechanism is **adjacent-atom selection under seed
variation**, not two estimator families, and the two published pairs are one
estimator reaching two neighbouring atoms. The constant-shift signature remains
the right test for a genuine family difference; it simply is not what this cell
shows. `contrast()` now returns `exact_lo`, `exact_hi` and `n_atoms` so the
distinction is visible without re-deriving it, and `ci_lo`/`ci_hi` are untouched
so nothing already published moves.

**The atom count here is 126, and an earlier draft of this report said 128.**
That was an error in this session's own code and is worth recording as one. The
number of distinct means of a size-`n` multiset drawn from `n` values is at most
`C(2n-1, n) = C(9,5) = 126`, so 128 was above its own ceiling. The cause: plain
`sum()` over `itertools.product` adds the same multiset in different orders, and
floating-point addition is not associative, so two multisets were split into
four values `5.551115123125783e-17` apart -- one ULP. Counting with
`math.fsum` on a canonically ordered tuple gives exactly 126, matching both the
combinatorial bound and the count independently on record for
`settled - softmax`.

The two contrasts agreeing at 126 is not a coincidence to explain away: 126 is
the generic count for any five distinct paired deltas, and both contrasts have
five distinct deltas. The earlier claim that the two figures were "different
lattices that do not conflict" was a rationalisation of a bug.

The percentile endpoints are unaffected -- they are order statistics over the
same 3,125 resamples and move by at most one ULP -- so the reconciliation above
stands unchanged.

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

## 5A. C1 `t*=32` — COMPLETE, and row G voids the whole rung

The crash killed the run at 13 of 15 units. `run_bucket` skipped the journalled
13 and the resume cost only the two missing `settledrow` seeds, exactly as
priced. `t*=32` is the rung where `t*` sits furthest above the arm's hop budget
of 2, so it is where the contract predicts hardest.

### Ceiling first, before this rung's numbers

Realised `sd_paired(settledrow - twinrow)` = **`0.000664`**. Seeds needed for a
95 % half-width below `RESOLUTION_13`: **1**. This run has 5, so the rung is
**adequately powered** — by a wide margin.

### The numbers

| cell | mean | |
|---|---|---|
| `softmax` | `1.003157` | at/above predict-the-mean |
| `twinrow` | `1.002587` | at/above predict-the-mean |
| `settledrow` | `1.003214` | at/above predict-the-mean |

**All three cells fail predict-the-mean.** Row G therefore voids every contrast
on this rung: a cell at or above `1.0` is credited nothing in either direction,
and a contrast between three failures licenses nothing.

| arm | vs | delta | CI | n+ | status |
|---|---|---|---|---|---|
| `settledrow` | `twinrow` | `-0.000627` | `[-0.001151, -0.000155]` | 0/5 | **VOID (row G)** |
| `twinrow` | `softmax` | `+0.000570` | `[+0.000088, +0.001052]` | 4/5 | **VOID (row G)** |
| `settledrow` | `softmax` | `-0.000057` | `[-0.000930, +0.000546]` | 3/5 | **VOID (row G)** |

Exact pairs, seed-free, over **126** atoms, are identical to the Monte-Carlo
pairs at every endpoint on this rung. The `settledrow`/`softmax` tie in the
house two-endpoint form: it **excludes a `softmax` advantage beyond `0.000930`
and a `settledrow` advantage beyond `0.000546`, and excludes nothing smaller.**

### Venus's prediction, scored on this rung

Her filing: *`settledrow - twinrow` reads inside `+-0.027260` at every rung with
every CI covering zero.* On `t*=32` it splits:

| half | reads |
|---|---|
| `\|delta\| < 0.027260` | **True** — `0.000627`, two orders inside |
| CI covers zero | **False** — `[-0.001151, -0.000155]` excludes zero, `n+ 0/5` |

`twinrow` beat `settledrow` on **all five seeds**. The magnitude half of her
prediction holds emphatically; the covers-zero half fails on this rung. **Both
halves are recorded, and neither is a verdict** — row G voids the contrast that
would have carried one, so this scores her prediction's arithmetic without
licensing any claim about settling.

### What this rung actually says

The resolution is superb (`sd_paired 0.000664`, one seed would have sufficed)
and it is spent on a contrast between three arms that all fail the absolute bar.
That is precisely the configuration row G exists to catch: a beautifully
resolved comparison of two failures. **No credit flows from `t*=32` in any
direction.**

## 5B. C1 `t*=8` - COMPLETE, and row G voids this rung too

### Ceiling first

Realised `sd_paired(settledrow - twinrow)` = **`0.000233`**, finer still than
`t*=32`'s. Seeds needed: **1**. This run has 5. **Adequately powered.**

### The numbers

| cell | mean | |
|---|---|---|
| `softmax` | `1.000933` | at/above predict-the-mean |
| `twinrow` | `1.000774` | at/above predict-the-mean |
| `settledrow` | `1.001047` | at/above predict-the-mean |

**All three fail predict-the-mean again. Row G voids every contrast.**

| arm | vs | delta | CI (MC) | exact | n+ | status |
|---|---|---|---|---|---|---|
| `settledrow` | `twinrow` | `-0.000272` | `[-0.000451, -0.000102]` | `[-0.000459, -0.000102]` | 1/5 | **VOID (row G)** |
| `twinrow` | `softmax` | `+0.000158` | `[-0.000051, +0.000380]` | `[-0.000051, +0.000380]` | 3/5 | **VOID (row G)** |
| `settledrow` | `softmax` | `-0.000114` | `[-0.000296, +0.000098]` | `[-0.000296, +0.000103]` | 2/5 | **VOID (row G)** |

Ties in the house two-endpoint form: `twinrow`/`softmax` **excludes a `softmax`
advantage beyond `0.000051` and a `twinrow` advantage beyond `0.000380`**;
`settledrow`/`softmax` **excludes a `softmax` advantage beyond `0.000296` and a
`settledrow` advantage beyond `0.000098`**. Neither excludes anything smaller.

**The lattice effect is live in this rung's own data.** Two of the three exact
pairs differ from their Monte-Carlo counterparts at one endpoint —
`settledrow`/`twinrow` at `ci_lo` (`-0.000459` exact against `-0.000451`
sampled) and `settledrow`/`softmax` at `ci_hi` (`+0.000103` against `+0.000098`)
— while the other endpoints coincide. That is exactly the adjacent-atom
selection recorded in section 2.5, now observed on a second corpus, and it is
why the exact pair is printed beside every interval rather than reasoned about
afterwards.

### Venus's prediction on `t*=8`

Identical split to `t*=32`:

| half | reads |
|---|---|
| `\|delta\| < 0.027260` | **True** — `0.000272`, two orders inside |
| CI covers zero | **False** — `[-0.000451, -0.000102]` excludes zero, `n+ 1/5` |

### The two completed rungs together

| rung | softmax | twinrow | settledrow | sd_paired | seeds needed |
|---|---|---|---|---|---|
| `t*=32` | 1.003157 | 1.002587 | 1.003214 | 0.000664 | 1 |
| `t*=8` | 1.000933 | 1.000774 | 1.001047 | 0.000233 | 1 |

Both rungs are **superbly resolved and entirely void**. Every cell on both fails
the absolute bar, and the means sit closer to `1.0` at `t*=8` than at `t*=32`.
On both, `twinrow` is the best of the three and `settledrow` the worst, and on
both the `settledrow - twinrow` interval excludes zero on the `twinrow` side —
but row G means none of that is creditable in either direction.

### Table status

**2 of 4 rungs complete - PARTIAL, NOT A READING.** `t*=2` is in flight,
`t*=1` queued. `verdict()` refuses quantified rows on incomplete data, so this
cannot be misread as a kill.

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
| M53 | `argmaxste` vs `twin` excludes a `twin` advantage beyond `0.029187` and an `argmaxste` advantage beyond `0.023107` | RUN | `-0.004092`, CI `[-0.029187, +0.023107]`, `n+ 2/5`. Two-endpoint form; the symmetric half-width `0.026147` understates the `twin` side and is withdrawn |
| M62 | **The two `argmaxste - argmax` pairs are one estimator reaching two adjacent atoms, NOT two estimator families** | RUN | Exact enumeration reproduces `[+0.212539, +0.245992]` from an independent implementation. Both Monte-Carlo endpoints sit ON the 126-atom lattice, each exactly one atom below its exact counterpart, and under seeds 0..99 they move independently: `ci_lo` takes 2 values, `ci_hi` takes 3 (`+0.2465162` appears). The equal `+1.056e-04` gaps are equal local atom spacing |
| M64 | **All three C1 cells at `t*=32` fail predict-the-mean; row G voids the rung** | RUN | `softmax` 1.003157, `twinrow` 1.002587, `settledrow` 1.003214, all >= 1.0 |
| M65 | The `t*=32` rung is adequately powered, and the power is spent on a void contrast | RUN | Realised `sd_paired(settledrow-twinrow)` `0.000664`; seeds needed 1, run has 5 |
| M66 | **Venus's prediction splits on `t*=32`** | RUN | `\|delta\| = 0.000627 < 0.027260` TRUE; CI `[-0.001151, -0.000155]` does NOT cover zero, `n+ 0/5`. Recorded, not adjudicated -- row G voids the contrast |
| M67 | The crash resume cost only the two missing units, as priced | RUN | Journal held 13 of 15; `run_bucket` skipped them and ran `settledrow` sd3 and sd4 only |
| M68 | **All three C1 cells at `t*=8` fail predict-the-mean; row G voids that rung too** | RUN | `softmax` 1.000933, `twinrow` 1.000774, `settledrow` 1.001047 |
| M69 | `t*=8` is even better resolved than `t*=32`, and equally void | RUN | Realised `sd_paired` `0.000233` against `0.000664`; seeds needed 1 on both |
| M70 | **The adjacent-atom effect is observed on a second corpus** | RUN | On `t*=8`, `settledrow-twinrow` exact `ci_lo` is `-0.000459` against sampled `-0.000451`, and `settledrow-softmax` exact `ci_hi` is `+0.000103` against sampled `+0.000098`; the other endpoints coincide |
| M71 | Venus's prediction splits identically on both completed rungs | RUN | `\|delta\|` inside `0.027260` on both; CI covers zero on neither |
| M63 | **The atom count is 126. This session first reported 128, which was its own bug** | RUN | `C(2n-1, n) = C(9,5) = 126` is the combinatorial ceiling, so 128 was impossible. Plain `sum()` over `itertools.product` split two atoms by one ULP, `5.551115123125783e-17`, because float addition is not associative. `math.fsum` on a canonically ordered tuple returns 126. Percentiles move by at most one ULP and the reconciliation is unaffected |
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
`argmaxste - twin` null excludes a `twin` advantage beyond `0.029187` and an
`argmaxste` advantage beyond `0.023107` and excludes nothing smaller, so
"indistinguishable" is a statement about what is ruled out, never about
identity; a true delta of `-0.029` is NOT ruled out. Whether a trained one-hot
selection also
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
