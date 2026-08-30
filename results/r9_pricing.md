# R9 pricing — what the remaining work costs

Produced in R9 iteration 1 by the runner role. **Every figure here is measured
or derived from a measurement with the arithmetic shown.** No figure is an
estimate from memory. A price is not a plan: this file says what things cost,
not what to buy.

## 0. The box, and why it is a term in every number

| | |
|---|---|
| OS / CPU | Windows 11, Intel Raptor Lake (Family 6 Model 183) |
| GPU | NVIDIA GeForce RTX 4060 Laptop, **8188 MiB** |
| torch | 2.5.1+cu121, CUDA available |
| threads | `torch.get_num_threads()` reads **20** before any `scale` import and **2** after |

The thread count is not a launcher setting. `torch.set_num_threads(2)` is pinned
inside `scale/arm_s.py:86` and eight sibling modules, with the in-file comment
*"pinned HERE, not by the launcher"*. Every arm timing in this repo — journalled
or fresh — is a 2-thread timing, and no amount of spare core count changes it.

**Contention band.** Two independent lanes were re-measured this session against
their own journal medians:

| lane | measured this session | journal median | ratio |
|---|---|---|---|
| CPU `softmax e3_t1` seed 5 | **30 s** unit, 33.4 s wall | 17.13 s | **1.75x** |
| CUDA `softmax e3_t2` seed 0 | **4 s** unit, 7.8 s wall | 1.59 s | **2.5x** |

Every extrapolation below is therefore given twice: at the journal-median rate
(a quiet box) and at **2.5x** (this box). The journal's own internal spread is
wider still — `settled e3_t1` reads min 264.73 s, median 379.32 s, max
**3004.14 s**, a 7.9x range across five seeds of one cell.

**Fixed cost per invocation** is 3.4 s (CPU) and 3.8 s (CUDA), measured as
`wall - unit` on the two runs above. It matters only for the CUDA softmax cells,
which cost less than the interpreter that starts them.

## 1. Measured base rates

### 1.1 CPU ladder — `results/m3_quintuple_v2.jsonl`

Median of 5 seeds, `s=64 d=24 steps=150 n_train=2048 n_eval=2048`, seconds:

| arm | `e3_t1` | `e3_t2` | `e3_t8` | `e3_t32` |
|---|---|---|---|---|
| `softmax` | 17.13 | 16.87 | 16.61 | 17.34 |
| `twin` | 179.99 | 35.96 | 56.05 | 84.31 |
| `settled` | 379.32 | 77.81 | 104.31 | 223.68 |

`e3_t1` is the expensive rung for the two ceq arms, not the cheap one. Cost
otherwise rises with `t*`, and `softmax` is flat across all four tasks.

### 1.2 CUDA lane — `results/m3_quintuple_v2_cuda.jsonl`

| arm | `e3_t1` | `e3_t2` | `e3_t8` | `e3_t32` |
|---|---|---|---|---|
| `softmax` | 1.59 | — | — | 1.58 |
| `twin` | 66.99 | — | — | — |
| `settled` | 66.44 | — | — | 69.10 (n=2) |

The task dependence that dominates the CPU lane has **collapsed** on the GPU:
`settled` reads 66.44 s at `t1` and 69.10 s at `t32`, where the same two cells
differ by 1.7x on CPU. The GPU lane is launch-bound, not arithmetic-bound, so
`e3_t2` and `e3_t8` are priced flat inside the measured `[66.4, 69.1]` bracket.

### 1.3 `foreman_looped` — the two journalled rows are not a price

`results/foreman_looped.jsonl` held two rows before this session, both `e3_t2`
at `steps=150`, reading **267.8 s** (`softmax`) and **296.3 s** (`looped3`).
A fresh `e3_t8` cell of the same arm, steps and sizes measured **74.9 s** this
session. In the quintuple lane every arm is *cheaper* at `e3_t2` than at `e3_t8`
(`twin` 35.96 vs 56.05, `settled` 77.81 vs 104.31), so those two rows are
inverted against the task's own cost curve by roughly 4x. They are contention
artifacts of the same kind as the 3004.14 s outlier in §0, and are not used
below.

### 1.4 The cost model: fixed cost plus per-step

Two cells of the same arm, task and sizes, differing only in `steps`, measured
this session:

| `steps` | seconds |
|---|---|
| 150 | 74.9 |
| 600 | 207.4 |

That is **2.77x for a 4x step increase**, not 4x. Solving the two-point system:

```
 74.9 = C + 150·P
207.4 = C + 600·P
  ->  P = 132.5 / 450 = 0.2944 s per step
  ->  C = 74.9 - 150(0.2944) = 30.7 s fixed
```

The fixed 30.7 s is not batch construction — the `e3_*` builders take **≤ 0.05 s
at n=4096**, and one `run_cell` pays ≤ 0.13 s of build in total. It is the
0-step control, the `n_eval=4096` eval forward, and the `n_boot=10000` bootstrap.
**A two-point fit cannot detect curvature**; treat `C` and `P` as a bracket, not
a law.

## 2. The three priced items

### 2.1 D1 — `foreman_looped`, **2 cells left of 6** (was 4)

Two of the four missing cells were run and journalled this session:

| task | arm | steps | eval NRMSE | seconds |
|---|---|---|---|---|
| `e3_t8` | `looped3` | 150 | 1.101899 | 74.9 |
| `e3_t8` | `looped3` | 600 | 1.447873 | 207.4 |

Remaining: `(e3_t2, softmax, 600)` and `(e3_t2, looped3, 600)`.

Price, from §1.4 at `steps=600`: **2 × 207 s = 415 s**, and that is a *this-box*
rate, not a quiet-box one, because both fit points were measured here. `e3_t2`
is the cheaper task than `e3_t8` for every arm in §1.1, so 415 s is an upper
bound.

> **D1 costs about one 420 s bucket.** It is the cheapest science on the board
> by a wide margin, and it is one bucket from `falsifier()` returning
> `complete: True`.

### 2.2 D2 — CUDA lane, **37 cells left of 60** (was 38)

The missing set, computed from the journal rather than from prose:

| arm | `e3_t2` | `e3_t8` | `e3_t32` | total |
|---|---|---|---|---|
| `softmax` | 4 | 5 | 0 | **9** |
| `settled` | 5 | 5 | 3 | **13** |
| `twin` | 5 | 5 | 5 | **15** |
| | | | | **37** |

At the §1.2 rates (`softmax` 1.59 s, `settled` 67.8 s, `twin` 66.99 s):

```
softmax   9 × 1.59  =    14 s
settled  13 × 67.8  =   881 s
twin     15 × 66.99 = 1,005 s
                      -------
                      1,900 s  = 4.5 buckets   (journal rate)
                      4,751 s  = 11.3 buckets  (this box, 2.5x)
```

> **D2 costs 4.5 to 11.3 buckets.** The whole missing GPU lane is under two
> hours even at the pessimistic rate. Thirty-eight cells sounds expensive and is
> not: the GPU collapses `settled` from 379 s to 66 s.

### 2.3 One full 5-seed 3-arm rung

`softmax + twin + settled`, 5 seeds, one task, `steps=150 n_train=2048`:

| rung | CPU, journal rate | CPU, this box (2.5x) | buckets (journal / 2.5x) |
|---|---|---|---|
| `e3_t1` | 2,882 s (48.0 min) | 7,205 s | 6.9 / 17.2 |
| `e3_t2` | 653 s (10.9 min) | 1,633 s | 1.6 / 3.9 |
| `e3_t8` | 885 s (14.7 min) | 2,212 s | 2.1 / 5.3 |
| `e3_t32` | 1,627 s (27.1 min) | 4,067 s | 3.9 / 9.7 |
| **all four** | **6,047 s (1.68 h)** | 15,118 s (4.2 h) | **14.4 / 36.0** |
| any rung, CUDA | 682 s (11.4 min) | 1,705 s | 1.6 / 4.1 |

### 2.4 Check against the record

`STATE.md:21` records *"Measured unit cost on the contended box 375.4 s /
379.3 s (settled, n_train=2048), so one rung is ~77 min and the four-rung ladder
~5 h."*

The unit cost reproduces exactly — the journal median for `settled e3_t1` is
**379.32 s**. The rung and ladder figures do not: 77 min is 12 cells priced at
the `settled` rate, and a rung is not 12 `settled` cells. It is 5 `settled` +
5 `twin` + 5 `softmax`, and `twin` is 2.1x cheaper and `softmax` 22x cheaper.
Measured, the four-rung ladder is **1.68 h, not 5 h — `STATE.md:21` over-prices
it by 3.0x.**

## 3. Round arithmetic

| constraint | value | source |
|---|---|---|
| foreground Bash call cap | 600 s | harness |
| `--budget` default | 420.0 s | `scale/bucket.py:126` |
| invocation overhead | 3.4 s CPU / 3.8 s CUDA | measured, §0 |
| iterations left | 14 of 15 | this is iteration 1 |

**Background invocations are not bound by the 600 s cap.** Two `foreman_looped`
runs of 79 s and 211 s completed detached this session, across turns. The
binding constraint for a long unit is the iteration's own wall clock, not the
call cap. This matters for exactly one cell class: a `settled e3_t1` CPU cell at
379 s fits a background call comfortably and a foreground one narrowly, and its
3004 s worst case fits neither foreground nor a bucket.

**Buckets per iteration is a free parameter and is not measured here.** At an
assumed 4 buckets per iteration, 14 iterations buy 56 buckets = 23,520 s = 6.5 h
of compute. Against that:

| | buckets (journal / 2.5x) | share of 56 |
|---|---|---|
| D1, 2 cells | 1.0 / 1.0 | 2% |
| D2, 37 cells | 4.5 / 11.3 | 8-20% |
| Full 4-rung CPU ladder | 14.4 / 36.0 | 26-64% |
| **all three** | **19.9 / 48.3** | **36-86%** |

D1 and D2 together are under 12 buckets even pessimistically. They fit several
times over.

## 4. What does not fit, plainly

### 4.1 Anything at `s=1024`. This is the hard one.

`impact` and `impact_hetero` are floored at `IMPACT_MIN_NODES = 1024`
(`scale/impact.py:73`) and are both registered in `M3_TASKS`
(`scale/negation_scope.py:1077,1081`). Two independent blockers, either one
fatal.

**Time.** Forward-plus-backward of the `settled` arm, batch n=64, 2 threads,
measured this session:

| `s` | ms/step | vs `s=64` |
|---|---|---|
| 64 | 6.45 | 1x |
| 256 | 48.74 | 7.6x |
| 1024 | 872.59 | **135x** |

The scaling exponent over the last leg is `log(17.9)/log(4) = 2.08` — quadratic,
as attention is. Applying the measured 135x ratio to the measured `settled e3_t1`
cell (379.32 s at `s=64`) gives **51,206 s = 14.2 hours for one cell, one seed,
one arm**. A 5-seed 3-arm rung is over 100 hours.

**Memory, and this one is absolute.** `scale/paired_arm.py:70-73` trains
**full-batch**: `model(x_train)` over all `n_train` rows on every step, with no
minibatching anywhere in the loop. At `n_train=2048, s=1024` a single attention
activation is

```
2048 × 1024 × 1024 × 4 bytes = 8.59 GB
```

which is **8192 MiB**, against the GPU's **8188 MiB total**. One activation is
larger than the whole card, and autograd holds it for the backward pass. It does
not fit before the clock matters. Cutting `n_train` to fit changes the shape
every published reading uses.

> **No IMPACT training cell fits this round at any iteration count.** The
> registration cannot be exercised as a training task without changing the
> training loop or the shape. This is a structural fact about
> `scale/paired_arm.py`, not a budget shortfall.

### 4.2 `e2_consequence` — blocked on correctness, not on cost

Never trained; zero rows in any `results/*.jsonl`. Its bar is broken at
`steps=150` and the cause is diagnosed and unfixed (`STATE.md:73-76`). Once the
bar is fixed, a 5-seed 3-arm rung at `steps=600` prices at the `e3_t2` rung rate
scaled by the §1.4 step factor: `653 × 2.77 = 1,809 s`, **4.3 buckets**. The
clock is not what is stopping it.

### 4.3 A second full CPU ladder is affordable but dominant

14.4 buckets at the journal rate, 36.0 at this box's. It fits, and it would
consume between a quarter and nearly all of an assumed 56-bucket round on its
own. The CUDA equivalent of the same four rungs is `4 × 682 = 2,728 s`, **6.5
buckets** — 2.2x cheaper overall.

That 2.2x hides the shape that matters. The GPU advantage is entirely in the
expensive rungs and is negative in the cheap one:

| rung | CPU / CUDA |
|---|---|
| `e3_t1` | **4.23x** cheaper on GPU |
| `e3_t32` | 2.39x cheaper on GPU |
| `e3_t8` | 1.30x cheaper on GPU |
| `e3_t2` | **0.96x — the GPU is marginally slower** (682 s vs 653 s) |

Because the GPU lane is launch-bound (§1.2) it charges a flat ~67 s for
`settled` and `twin` regardless of task, while CPU `settled` at `e3_t2` costs
only 77.81 s. Moving the cheap rungs to the GPU buys nothing.

## 5. What this file does not price

Only the `s=64 d=24 n_train=2048 n_eval=2048` shape. Nothing at `n_train=8192`
beyond the single observation that `softmax` reads 102.67 s there against
17.13 s at 2048 with `n_eval` also changing from 512 to 2048, so that 6.0x
factor is confounded across two variables and is not used. The CUDA prices for
`e3_t2` and `e3_t8` are interpolated inside a measured `[66.4, 69.1]` bracket
and no GPU `settled` or `twin` cell has actually run at either task. The
`s`-scaling ratios in §4.1 come from a batch of 64 and a synthetic loss, not
from a real cell; the ratio is the load-bearing quantity and the absolute
ms/step figures are not. The §1.4 cost model rests on two points of one arm on
one task. And every second in this file was measured on one contended laptop
whose own journal shows a 7.9x spread across five seeds of a single cell.
