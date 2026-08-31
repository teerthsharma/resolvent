# V15 NEPTUNE — THE SYSTEMS LINE

Memory, wall clock and dispatch for the v15 deciding cells. Round 11, it.4.
Repo at `089e789`. Probe: `scripts/v15_neptune_sizing_probe.py`, output pasted
verbatim in §7.

**Provenance tag on every number.** `[MEASURED]` = read off this box this
session. `[MODULE]` = computed by a named function in `ceq/sizing.py`.
`[FITTED]` = projected from a law fitted here, with its R². `[INHERITED]` =
carried from an earlier round, cited by file and line. `[ASSUMED]` = a value the
contract does not fix, chosen here with a reason. `NOT MEASURED` where it could
not be obtained.

---

## 0. THE VERDICT IN FIVE LINES

1. **R2 at `n = 32768` FITS.** Measured peak **4.000 GiB** against **6.939 GiB**
   free VRAM — 2.61 GiB of headroom, 38% of free. It is not a projection: the
   allocation was made and the step was timed. `[MEASURED]`
2. **The calibrated model holds, and it holds CONSERVATIVELY.** measured /
   predicted = **0.950** at `n = 512/1024/2048` and **0.949** at `n = 32768` — a
   constant ratio across a 64x span in `n`. The module over-predicts by 5%; it
   never under-predicts. `[MEASURED]`
3. **The inherited timing exponent does not reproduce.** `secs ~ n^1.338`
   `[INHERITED]` re-measures here as **`n^1.1920`**, R² 0.999642 over five
   points `[MEASURED/FITTED]`. R2 is priced on the re-measured law, per L-TIME.
4. **The premise "the `[n,s,s]` operator is the dominant term" is FALSE at
   BED-M's shape.** At `s = 64, d_model = 16`, fp32, the operator term is
   **43.7%** of measured peak; the residual/MLP chain is 56.3%. The operator
   only dominates above `s = 74` at fp32. §2.4. `[MEASURED]`
5. **The it.9–it.23 plan costs 47.0 GPU-hours or 637 CPU-hours** at the full
   9600-step ladder over 184 cells `[FITTED]`. Nothing needs cutting *if the
   cells move to the CUDA device that sat idle while the last loop died of
   unaffordability*. On CPU-only, cut R3 and the step ladder — §6.

---

## 1. THE BOX

| quantity | value | how |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, sm_89, 24 MPs | `torch.cuda.get_device_properties(0)` `[MEASURED]` |
| VRAM total | 8,585,216,000 B = **7.996 GiB** | `torch.cuda.mem_get_info()[1]` `[MEASURED]` |
| VRAM free | 7,451,181,056 B = **6.939 GiB** | `torch.cuda.mem_get_info()[0]` `[MEASURED]` |
| VRAM already held | 1.057 GiB (display + other processes) | total − free `[MEASURED]` |
| host RAM | 15.71 GiB total, **6.49 GiB available** at probe start (58.7% used) | `psutil.virtual_memory()` `[MEASURED]` |
| host RAM at the R2 attempt | 5.293 GiB available | `psutil` inside §7 F `[MEASURED]` |
| CPU | 28 logical, torch default 20 threads; probe pinned to 12 | `os.cpu_count()` / `torch.get_num_threads()` `[MEASURED]` |
| torch | 2.5.1+cu121, numpy 1.26.4, Windows 11 | `[MEASURED]` |

The GPU is the same RTX 4060 Laptop that `ceq/sizing.py`'s docstring names as
the box the constants were calibrated on, so this is a re-validation on the
calibration hardware at a **different workload shape**, not on different silicon.

---

## 2. THE MEMORY MODEL, INSTANTIATED

### 2.1 The cell shape, and where each dimension came from

The v15 deciding cells are BED-M cells. BED-M is the chain corpus already in the
repo, and its cell shape is fixed by executable lines, not by the contract:

| symbol | value | source |
|---|---|---|
| `s` (sequence) | **64** | `scale/r10_capacity_sweep.py:76` — `S, D = 64, 24`, "the shape every e3 row in `results/m3_capability.txt` uses" `[MEASURED from source]` |
| `d_model` | **16** | `scale/m3_capability.py:83` — `D_MODEL = 16  # fixed per task spec` |
| MLP hidden | **128** | `scale/m3_capability.py:84` — `HIDDEN = 128`, i.e. an 8x MLP, not 4x |
| layers | **1** | `scale/m3_capability.Arm` is one operator + one MLP + one readout; there is no stack |
| heads | **1** | `bench._softmax_operator(q, k)` returns `[n, s, s]`; no head axis exists |
| batch | **= `n`** | `train_with_checkpoints` calls `model(x_train)` on the whole set: FULL BATCH. This is why `n` is a memory axis at all |
| dtype | **fp32** | no autocast anywhere in `scale/`; `torch.randn` default. bf16 is priced in §2.4 as an option, not as the status quo |
| readout width | **1** | `nn.Linear(d_model, 1)` — there is no vocab head |
| `d = 24` | distractor distance, **not** a tensor dim | `make_equilibrium_batch(n, s, d, ...)` positions the payload; it never sizes a tensor |

`n` is the number of TRAINING EXAMPLES, and because training is full-batch it is
also the batch dimension. That single fact is what makes `n = 32768` a memory
question at all, and it is what the contract's `n` means at R1/R2/R3.

### 2.2 The two known reasons a transferred constant could break

`ceq/sizing.py`'s constants were fitted on `ceq.lm.TinyLM`, `B=4 d=256 L=4 H=4`,
a `[B,H,S,S]` operator and a `[B,S,V]` cross-entropy head (module docstring,
lines 13–22). Two mismatches are declared before the numbers, not after:

1. **`C_RESIDUAL = 18`** counts activation elements per token per layer for a
   standard block with a **4x** MLP. The BED-M arm has an **8x** MLP
   (`HIDDEN = 128` against `d_model = 16`), and `activation_bytes` has no
   `d_ff_mult` term at all — the parameter enters `params()` and
   `flops_per_token()` only. So `C_RESIDUAL` is being applied outside its fit.
2. **`arm="softmax"` is the wrong call.** In `ceq/sizing.py`, "softmax" means
   the FUSED SDPA path, which forms no `[S,S]` tensor and whose activation
   memory is O(S). `scale/m3_capability.Arm`'s softmax branch calls
   `ceq.bench._softmax_operator`, which materialises `[n,s,s]` explicitly and
   hands it to autograd. On the memory axis the BED-M *softmax* arm is a
   *signed* arm. Calling `activation_bytes(arm="softmax")` here would drop the
   entire operator term and under-predict by exactly the quantity this node
   exists to price. Every prediction below uses **`arm="signed"`**.

### 2.3 R1 — BED-M, `t* = 2`, `n = 2048`

`sizing.activation_bytes(Config(d_model=16, n_layers=1, n_heads=1, d_head=16, vocab=1, seq=64), batch=2048, arm="signed", dtype="fp32")` `[MODULE]`

```
residual  C_RESIDUAL * L * n * s * d * rb = 18  * 1 * 2048 * 64 * 16 * 4.0 = 150,994,944 B = 0.141 GiB
operator  C_OPERATOR * L * H * n * s * s * ob = 3.9 * 1 * 1 * 2048 * 64 * 64 * 4.0 = 130,862,285 B = 0.122 GiB
head      C_HEAD * n * s * V * 4             = 2   * 2048 * 64 * 1 * 4       =   1,048,576 B = 0.001 GiB
--------------------------------------------------------------------------------------------------
predicted total                                                              = 282,905,805 B = 0.263 GiB
MEASURED  torch.cuda.max_memory_allocated() - base                           =                 0.250 GiB
ratio measured/predicted                                                     = 0.950
```

Plus `x_train` resident: `2048*64*16*4 = 8,388,608 B = 0.008 GiB`. R1's working
set is **0.258 GiB**. It fits on anything in this building.

### 2.4 R2 — BED-M, `t* = 8`, `n = 32768` — THE CELL THAT MATTERS

```
residual  18  * 1 * 32768 * 64 * 16 * 4.0     = 2,415,919,104 B = 2.250 GiB
operator  3.9 * 1 * 1 * 32768 * 64 * 64 * 4.0 = 2,093,796,556 B = 1.950 GiB
head      2   * 32768 * 64 * 1 * 4            =    16,777,216 B = 0.016 GiB
------------------------------------------------------------------------------
predicted total                               = 4,526,492,876 B = 4.216 GiB   [MODULE]
x_train   32768 * 64 * 16 * 4                 =   134,217,728 B = 0.125 GiB
predicted working set                                            4.341 GiB
MEASURED peak activations                     =                 4.000 GiB   [MEASURED]
ratio measured/predicted                      = 0.949
measured working set (peak + x_train)                            4.125 GiB
```

**THE `[n,s,s]` OPERATOR IS NOT THE DOMINANT TERM AT THIS SHAPE.** The brief
asserts it is; the arithmetic says otherwise and the measurement confirms the
arithmetic. Per token, the two terms are `C_RESIDUAL * d * rb = 18*16*4 = 1152`
bytes against `C_OPERATOR * s * ob = 3.9*s*4 = 15.6 s` bytes. They cross at
**`s = 73.8`**. BED-M runs at `s = 64`, below the crossover, so:

| term | predicted | back-solved from measurement | share of measured peak |
|---|---|---|---|
| residual / MLP chain | 2.250 GiB | 2.240 GiB | **56.3%** |
| `[n,s,s]` operator | 1.950 GiB | 1.750 GiB | **43.7%** |

Under `dtype="bf16_autocast"` the crossover moves to `s = 47.8`
(`18*16*2.2 = 633.6` against `3.9*3.4*s = 13.26 s`) and the operator *would*
dominate at `s = 64` — but nothing in `scale/` uses autocast, so bf16 is an
option to be taken, not a description of the current cell. Under bf16 the R2
prediction falls to `18*32768*64*16*2.2 + 3.9*32768*64*64*3.4 = 1,329,205,248 +
1,779,727,073 = 3.109 GiB`, a 26% saving that is **NOT MEASURED** here (bf16
autocast on a 16-dim model would also change the NRMSE, which is a capability
question and L-LEAN territory, not a systems one).

### 2.5 R3 — BED-K, and the assumption the contract does not fix

**The contract does not state `n` for R3.** PART IV names R1's `n = 2048` and
R2's `n = 32768` and then gives R3 only its acceptance criteria ("scan-only
`>= 0.95`, attention / fractional head near 0, COMPOSED arm matches the native
primitive", "`alpha-hat` recovered, `H-hat = alpha-hat + 1/2` within CI").
BED-K itself does not exist yet — it is built at `n4`/it.12–14.

`[ASSUMED]` **R3 runs at `s = 64`, `n = 32768`, fp32** — the same cell shape as
R2. Reason: R4 is "the composed arm within `Delta_res` of the native skyline on
BOTH BED-M and BED-K". A resolution statement comparing two beds at different
`n` compares two different resolutions, and `Delta_res = t_{.975,N-1} sd/sqrt(N)`
is a function of the seed sd, which is a function of `n`. Matching `n` across
the two beds is the cheapest way to make R4's sentence mean anything.

Under that assumption R3's memory is **R2's memory: 4.000 GiB measured per cell**,
for any arm whose retained-operator count matches the single-operator arm.

**THE ONE RISK, PRICED.** If BED-K's power-law bed needs a longer sequence — and
a Hurst estimator on 64 points is a thin read — the affordable `n` collapses
quadratically. Measured bytes per training example at `n = 256`, fp32, CUDA:

| `s` | measured act. B/example | + `x_train` B/example | largest `n` at 20% margin |
|---|---|---|---|
| 64 | 131,246 | 135,342 | **42,774** |
| 128 | 299,008 | 307,200 | 18,844 |
| 256 | 1,081,423 | 1,097,807 | 5,273 |
| 512 | 4,259,919 | 4,292,687 | 1,348 |

`[MEASURED]` for the per-example column (§7 B2), `[MEASURED+computed]` for the
`n` column: budget = `(6.939 − 0.200) GiB × 0.80`, where 0.200 GiB is held back
for `x_eval` at `n_eval = 4096` and its no-grad forward.

**So: if BED-K needs `s = 256`, R4's matched-`n` requirement caps BOTH beds at
`n ≈ 5,273`, and R2's `n = 32768` becomes unmatchable.** That is a design
decision for it.12–14 and it is cheaper to take it now than at it.15.

**NOT MEASURED:** the composed arm's own `C_OPERATOR`. The S-M carrier
(`W_ij = exp(C_i − C_j)` on values) and the S-K memory kernel are two operator
families in one forward; the arm may retain 6–8 `[n,s,s]` tensors rather than
3.5. Every number in §2.4–§2.5 is for a **single-operator** arm and must be
re-measured with this same probe once ARM PL exists at it.6–7. Under a doubled
`C_OPERATOR`, R2's peak goes to `2.240 + 3.500 = 5.74 GiB`, which still fits at
6.939 GiB free but leaves only 1.0 GiB of margin.

---

## 3. VALIDATION — THE MODEL AGAINST THE ALLOCATOR

A predicted number from an uncalibrated formula is worth nothing; that is how
the old line came to be 6.63x low (`3.9 × 3.4/2.0 = 6.63`, `V13_CLAIM_AUDIT.md:59`).
Five real allocations, fp32, CUDA, peak by `torch.cuda.max_memory_allocated()`
above a baseline taken after the model, the inputs and the mask cache exist.

### 3.1 The `n` sweep — and why it licenses the extrapolation

| `n` | predicted (GiB) `[MODULE]` | measured (GiB) `[MEASURED]` | measured/predicted |
|---|---|---|---|
| 256 | 0.033 | 0.039 | 1.191 |
| 512 | 0.066 | 0.063 | **0.950** |
| 1,024 | 0.132 | 0.125 | **0.950** |
| 2,048 | 0.263 | 0.250 | **0.950** |
| **32,768** | **4.216** | **4.000** | **0.949** |

**VERDICT: the calibrated model HOLDS.** Both terms of `activation_bytes` are
exactly linear in `n` at fixed `s`, so a ratio measured at `n = 512` *is* the
ratio at `n = 32768` — and it is: 0.950 against 0.949, across a 64x span. The
model over-predicts by 5.1%, which is the right direction for a gate.

The `n = 256` row is the one outlier at 1.191, and it is an allocator artefact,
not a model failure: measured 41.9 MB against predicted 34.9 MB is a ~7 MB fixed
overhead (cuBLAS workspace + block granularity) that is 20% of a 35 MB working
set and 0.16% of a 4 GiB one. It is reported rather than dropped, and it is the
reason the through-origin fit in §7 G starts at `n = 512`.

**No new correction factor is needed.** If one is wanted as a point estimate,
multiply `activation_bytes` by **0.95** at this shape; keeping 1.00 costs 5%
of headroom and buys a guaranteed upper bound, which is what a gate should have.

### 3.2 The `s` sweep — separating the two terms

Varying `s` at fixed `n` is what actually tests `C_OPERATOR`, because it is the
only axis on which the two terms scale differently (`s` against `s²`).

| `s` | predicted (GiB) | measured (GiB) | measured/predicted |
|---|---|---|---|
| 64 | 0.0329 | 0.0313 | 0.950 |
| 128 | 0.0963 | 0.0713 | 0.740 |
| 256 | 0.3146 | 0.2578 | 0.820 |
| 512 | 1.1166 | 1.0156 | 0.910 |

Back-solving the two constants exactly from the `s = 64` and `s = 512` rows
(two equations, two unknowns, `C_HEAD` subtracted at its module value):

```
C_RESIDUAL   module 18.00   this shape 17.92   (−0.4%)
C_OPERATOR   module  3.90   this shape  3.50   (−10.3%)
```

`C_RESIDUAL = 18` transfers almost exactly despite the 8x MLP — a coincidence of
this shape (`2 LN + q + k + a@x + z + h_pre + h_post + out ≈ 21 d`, against 18),
not a law, and it should be re-checked if `d_model` or `HIDDEN` move.

`C_OPERATOR = 3.5` rather than 3.9 is the real finding, and it is explicable
from the source: `_softmax_operator` builds `w = q@kᵀ/√d` → `masked_fill` →
`softmax` → `masked_fill`, and autograd retains the softmax output and the final
masked-fill output while `w` and the first masked-fill are transient but land
inside the same peak. Three-and-a-half is a peak-overlap count, not an integer,
and the LM's 3.9 includes the extra retention a stacked block adds.

**The two-constant model is not exact across `s`:** it over-predicts by 26% at
`s = 128` and 18% at `s = 256`. That is allocator block-rounding and workspace
reuse, and it is why §2.5's `s`-scaling table uses the **measured** bytes per
example at each `s` rather than the model. Direction of error is uniformly
conservative — the module never under-predicted at any `(n, s)` probed.

---

## 4. WALL CLOCK VERSUS FLOP

### 4.1 What the inherited number actually is

`V13_DAG_TASKLIST.md:88-91` records, on an idle host at `threads = 12,
steps = 150, t* = 8`: medians `n=2048 → 18.19 s`, `n=8192 → 94.74 s`,
`n=16384 → 293.94 s`, "a log-log slope of `secs ~ n^1.338`". `V13_CLAIM_AUDIT.md:272-278`
already struck the word "fitted": **1.33813 is a two-endpoint slope**; three-point
OLS on the same data is **1.31700**; the adjacent pairwise slopes are 1.1904 and
1.6335, so the inherited relation is convex over its own measured range.

L-TIME requires the citing iteration to re-measure. Done.

### 4.2 The re-measurement

Median of 12 timed steps (fwd + bwd + `Adam.step`), 2 warm-up steps discarded,
`torch.set_num_threads(12)` to match the inherited pin, `s = 64`, random data.
Median not mean: on a laptop the scheduler donates occasional 3x outliers.

| `n` | CPU s/step `[MEASURED]` | ×150 steps | CUDA s/step `[MEASURED]` | ×150 | CPU/GPU |
|---|---|---|---|---|---|
| 2,048 | 0.0810 | 12.2 s | 0.0110 | 1.7 s | 7.3x |
| 4,096 | 0.1792 | 26.9 s | 0.0201 | 3.0 s | 8.9x |
| 8,192 | 0.4246 | 63.7 s | 0.0403 | 6.0 s | 10.5x |
| 16,384 | 0.9922 | 148.8 s | 0.0804 | 12.1 s | 12.3x |
| 32,768 | 2.1432 | 321.5 s | 0.1605 | 24.1 s | 13.4x |

**Fitted laws, log-log OLS over all five points:**

| device | exponent | R² | endpoint slope |
|---|---|---|---|
| CPU, 12 threads | **1.1920** | **0.999642** | 1.1814 |
| CUDA | **0.9734** | **0.999316** | 0.9614 (4-pt) |

`[FITTED]`. Four-point fits excluding `n = 32768` give CPU 1.2088 (R² 0.999688)
and CUDA 0.9602 (R² 0.998844); the `n = 32768` anchor moves the CPU exponent by
0.017, so the law is stable rather than being carried by its endpoint.

### 4.3 It differs from 1.338, and by how much

**The inherited exponent does not reproduce on this box: 1.1920 against 1.338,
a difference of 0.146.** Two things are true about the gap and both are stated:

- The re-measured relation is **still convex** — pairwise slopes 1.147
  (2048→8192), 1.295 (8192→16384), 1.109 (16384→32768) — but far less so than
  the inherited 1.190 / 1.634. The inherited 1.634 tail at `n = 16384` is the
  entire source of its higher endpoint slope, and the same tasklist entry warns
  "if page-file pressure begins above a ~2 GB working set the true slope is
  worse". A 2 GiB working set is exactly `n = 16384` at this shape (§2.4 scaled).
  The inherited tail is consistent with memory pressure on a 16 GiB host; this
  measurement has the same host with 6.49 GiB free and does not show it.
- **The absolute per-step costs are 1.5–2.0x faster here** than the inherited
  numbers imply (`18.19/150 = 0.1213` against 0.0810 at `n = 2048`;
  `293.94/150 = 1.9596` against 0.9922 at `n = 16384`). The inherited `secs`
  field includes two extra full forwards and a bootstrap per rung, but that is a
  near-constant additive term that would *flatten* a log-log slope, not steepen
  it, and it cannot account for a 1.5x ratio at the smallest `n`. **The residual
  cause is NOT DIAGNOSED.** Since it is a level shift and not a slope, the
  projections below anchor on the freshly measured points and not on the
  inherited ones.

**Under L-TIME, 1.338 is retired for this round.** Every projection below uses
`n^1.1920` (CPU) and `n^0.9734` (CUDA), both R² > 0.999.

### 4.4 R1 and R2 at `N = 8` seeds

Contract fixes `N = 8`. Step budget is the open dial: `scale/r10_capacity_sweep.py:81`
runs `STEP_RUNGS = (150, 600, 2400, 9600)`, nested — a 9600-step run passes
through all four rungs, so the full ladder costs 9600 steps and yields four
readings. The last round bought only the 150 rung at `n = 8192` and journalled
the rest `"unaffordable, see priced DAG"`
(`results/r10_r10_it21_t8_n8192_capacity_softmax_t8.jsonl:8`).

Per **arm**, `N = 8` seeds, both step budgets:

| cell | device | 150 steps | full 9600 ladder |
|---|---|---|---|
| R1 `n=2048` | CPU | 0.027 h | 1.73 h |
| R1 `n=2048` | CUDA | 0.0037 h | 0.23 h |
| R2 `n=32768` | CPU | 0.714 h `[MEASURED]` | **45.72 h** `[MEASURED]` |
| R2 `n=32768` | CUDA | 0.054 h `[MEASURED]` | **3.42 h** `[MEASURED]` |

Tagged `[MEASURED]` because the `n = 32768` step time was timed at full size,
not extrapolated to it. **The fitted law over-prices that cell by 6%** — it
projects 2.2715 s/step against the measured 2.1432 on CPU, and under-prices the
CUDA cell by 4% (0.1538 projected against 0.1605 measured). §6's tables use the
fitted law throughout so the nodes are priced on one consistent rule; where a
number here and a number there differ by a few percent at `n = 32768`, that is
the gap and its direction.

**R1 + R2 total at `N = 8`, two trained arms each** (ARM PL and softmax; the
scan skyline is a computed reference column, not a fourth training run —
`V13_DAG_TASKLIST.md:75-78`):

| | CPU | CUDA |
|---|---|---|
| 150-step floor | **1.48 h** | **0.115 h** |
| full 9600 ladder | **94.9 h** | **7.30 h** |

FLOP is not the binding constraint and is not offered as one. `sizing.flops_per_token`
models an LM, not this arm, and the arm is bandwidth-bound on a materialised
`[n,s,s]`, not compute-bound: the CUDA exponent of **0.9734** is essentially
linear in `n`, which is the signature of a memory-traffic-bound kernel, while a
compute-bound `s²`-dense operator at fixed `s` would also be linear. The
distinguishing number is the CPU exponent of 1.192, which is *super*linear and
therefore is not FLOPs either — it is cache-residency loss as the working set
outgrows L3. **The wall-clock law is the price, and FLOP-per-token is not quoted
as a substitute for it.**

---

## 5. THE AFFORDABILITY VERDICT FOR R2

**R2 at `n = 32768` FITS on this GPU, measured.**

```
free VRAM                                           6.939 GiB = 7,451,181,056 B   [MEASURED]
predicted working set (activations + x_train)       4.341 GiB                     [MODULE]
MEASURED peak activations                           4.000 GiB                     [MEASURED]
x_train resident                                    0.125 GiB
eval-set reserve (x_eval @ n_eval=4096 + fwd)       0.200 GiB                     [ASSUMED, see below]
------------------------------------------------------------------------------
total                                               4.325 GiB
headroom                                            2.614 GiB  = 38% of free      [MEASURED]
```

The 0.200 GiB eval reserve is `[ASSUMED]`: `x_eval` at `n_eval = 4096` is
16.8 MB and its no-grad forward peaks at roughly one `[4096,64,64]` operator
(67 MB) plus the residual chain, so 0.200 GiB is a rounded-up bound rather than
a measurement. Removing it entirely moves `n_max` by 1,589.

**Where it stops fitting** — measured bytes per training example, through-origin
least squares over the measured peaks at `n = 512…32768` (§7 G):
`activations 131,086 + x_train 4,096 = 135,182 B/example`.

| safety margin | budget | largest affordable `n` |
|---|---|---|
| 0% | 6.687 GiB | 53,112 |
| 10% | 6.018 GiB | 47,800 |
| **20%** | **5.349 GiB** | **42,489** |
| 30% | 4.681 GiB | 37,178 |

**The largest `n` this GPU affords at a 20% margin is `n = 42,489`.** R2's
`n = 32768` sits at 77% of that, which is why the answer is a fit rather than a
squeeze. The next octave, `n = 65536`, needs 8.26 GiB and **does not fit** —
it exceeds even total VRAM. **`n = 32768` is the last power of two that runs on
this device**, and that is the honest ceiling to write into the DAG.

**On CPU R2 also fits, but with no margin worth having.** Working set 4.341 GiB
against 5.293 GiB available at the time of the attempt — a 0.95 GiB cushion on a
16 GiB host at 59% load. The step was timed successfully (2.1432 s/step), so it
runs; but a browser tab decides whether the next one does. The tasklist's note
that "Windows commits beyond available-physical" (`V13_DAG_TASKLIST.md:105-110`)
means the failure mode is not an exception, it is page-file thrash that silently
inflates the wall clock — which is precisely the mechanism that produced the
inherited 1.634 tail slope. **The CPU route to R2 is feasible and fragile; the
CUDA route is feasible with 38% headroom.**

---

## 6. THE DISPATCH LINE — it.9 THROUGH it.23

### 6.1 The cell count

A **cell** is one `(arm, bed, t*, n, seed)` training run, read at whatever step
rungs it buys. `N = 8` seeds is contract-fixed. Arms-per-node is not, so each
assumption is named:

| node | contract text | cells | `n` | assumption |
|---|---|---|---|---|
| it.9 R1 | "ARM PL vs softmax vs scan-skyline" | 2 arms × 8 = **16** | 2,048 | skyline is a computed column, not a trained arm `[V13_DAG_TASKLIST.md:75-78]` |
| it.10 R2 | same two arms at `t*=8` | 2 × 8 = **16** | 32,768 | — |
| it.11 | "the linear-probe diagnosis" | 1 × 8 = **8** | 2,048 | fires only on non-crossing; priced anyway |
| it.12-14 | "scan skyline + composed arm" | 1 × 8 × 2 = **16** | 2,048 / 32,768 | composed arm on BED-M at both deciding scales; skyline + BED-K registration are non-training |
| it.15-17 R3 | scan-only / attention / fractional head / composed, on delay + power-law beds | 4 × 2 × 8 = **64** | 32,768 `[ASSUMED §2.5]` | R4 is a table over R1–R3 and adds no cells |
| it.18-21 R5 | BED-1 S5' exit accuracy + control | 2 × 8 = **16** | 32,768 `[ASSUMED]` | CK test, Morse census, Pesin deficit are analyses of the trained artifacts |
| it.22-23 R6 | "`zeta > 0` vs `zeta = 0` at fixed n" | 2 × 8 × 3 = **48** | 512 / 2,048 / 8,192 | the claim is "2x fewer examples to R1's crossing", which is an `n`-sweep, so three rungs below R1's `n` |
| | | **184 cells** | | |

### 6.2 The summed wall clock

`[FITTED]` from `n^1.1920` (CPU) and `n^0.9734` (CUDA), both R² > 0.999.

**150-step floor — the budget the last loop could actually buy:**

| node | cells | `n` | CPU h | CUDA h |
|---|---|---|---|---|
| it.9 R1 | 16 | 2,048 | 0.05 | 0.007 |
| it.10 R2 | 16 | 32,768 | 1.46 | 0.105 |
| it.11 diagnosis | 8 | 2,048 | 0.03 | 0.004 |
| it.12-14 composed @R1 | 8 | 2,048 | 0.03 | 0.004 |
| it.12-14 composed @R2 | 8 | 32,768 | 0.73 | 0.053 |
| it.15-17 R3 | 64 | 32,768 | 5.85 | 0.421 |
| it.18-21 R5 | 16 | 32,768 | 1.46 | 0.105 |
| it.22-23 R6 | 48 | 512/2048/8192 | 0.34 | 0.036 |
| **TOTAL** | **184** | | **9.96 h** | **0.734 h** |

**Full 9600-step ladder — the budget that produces converged readings:**

| node | cells | CPU h | CUDA h |
|---|---|---|---|
| it.9 R1 | 16 | 3.43 | 0.453 |
| it.10 R2 | 16 | 93.60 | 6.734 |
| it.11 diagnosis | 8 | 1.72 | 0.227 |
| it.12-14 composed @R1 | 8 | 1.72 | 0.227 |
| it.12-14 composed @R2 | 8 | 46.80 | 3.367 |
| it.15-17 R3 | 64 | 374.39 | 26.935 |
| it.18-21 R5 | 16 | 93.60 | 6.734 |
| it.22-23 R6 | 48 | 22.02 | 2.318 |
| **TOTAL** | **184** | **637.28 h** | **46.99 h** |

### 6.3 Is it plausible, and what to cut

**637 CPU-hours is 26.5 days of continuous compute. It is not plausible** and it
is the same wall the last loop hit, one octave further out. **47.0 GPU-hours is
two days**, which is an AFK loop run, and every cell in it fits in VRAM with
38% headroom at the largest one.

**THE PRIMARY RECOMMENDATION IS NOT A CUT.** It is that the deciding cells move
to the CUDA device. `scale/m3_capability.py`'s docstring says **"CPU ONLY"** and
`scale/r10_capacity_sweep.py` has no `--device` flag; the RTX 4060 sat idle
through the entire round that died of unaffordability. The change is small —
an argparse flag, `.to(device)` on the model and the two batches — but it is a
**code change to an existing module, which this node is forbidden to make**, so
it is filed as a required predecessor to it.9 rather than performed.

**Three conditions attach to that move, and none of them is optional:**

1. **A CUDA cell may never be pooled with a CPU-taken cell.** M-10 already
   records that thread count alone moves NRMSE by 0.002345, which is 0.464 of
   the equivalence margin `Δ_eq = 0.005051` at `t*=2, n=2048`
   (`V13_DAG_TASKLIST.md:79-84`). A device change alters the reduction order far
   more than a thread change does. Every published R10 number is CPU-taken.
2. **The instrument must be re-RED-gated on CUDA before any cell is credited:**
   `calibrate_bar` and the 0-step gate (`red["ok"]`, `GATE_TOL = 1e-3`) re-run on
   the device, journalled with `device` in the header record alongside `threads`.
3. **`torch.use_deterministic_algorithms` or an explicit non-determinism note.**
   cuBLAS split-K reductions are run-to-run nondeterministic; the 8-seed sd is
   the instrument's resolution and must not absorb kernel noise.

**If the CUDA move is refused, here is the cut list, in order.** It is ordered
by hours-saved-per-claim-lost, not by convenience:

Savings are **cumulative and applied in order**, each measured against the plan
as the previous cut left it — not against the 637.28 h baseline, which would
double-count.

| # | cut | CPU h before → after | saves | what is lost |
|---|---|---|---|---|
| 1 | **R3 at `n = 4096` instead of 32,768** | 374.39 → 31.39 | **343.00 h** | Nothing R3 claims. R3's acceptance criteria are a *qualitative separation* — "scan-only ≥ 0.95, attention / fractional head near 0". A 0.95-against-0.00 gap does not need R2's resolution; it needs a bed where the separation exists at all. **This is the single largest line in the plan and the cheapest to cut.** Cost: R4's matched-`n` two-sided table must then be read at `n = 4096` on both beds, which means R2's `n = 32768` is a BED-M-only cell and does not enter R4. |
| 2 | **R5 (BED-1) at `n = 8192`** | 93.60 → 17.93 | **75.67 h** | Nothing stated. R5's reading is exit accuracy under committor labels plus a CK test; the CK test `T̂(nτ) ≈ T̂(τ)ⁿ` is a *consistency* check on a transition operator, not a scaling reading. |
| 3 | **Drop R6's `n = 8192` rung** | 17.93 → 0 | **17.93 h** | One point of the "2x fewer examples" curve. Keep 512 / 2048; the claim is that `ζ > 0` reaches R1's crossing at half the examples, and R1's `n` is 2048, so a rung at 4x R1's `n` proves nothing about halving it. |
| 4 | **Cap the ladder at 2400 for every node except R1 and R2** | 103.65 → 25.91 | **77.74 h** | The 9600 rung on the non-deciding nodes. Justified: R1/R2 are *floor-crossing* verdicts where the last rung is the one that decides whether the threshold is crossed; R3/R5/R6 are *separation* verdicts where 2400 steps already resolve a 0.95-against-0.00 gap. |

**Trimmed plan, cell by cell:**

| node | cells | `n` | steps | CPU h |
|---|---|---|---|---|
| it.9 R1 | 16 | 2,048 | 9,600 | 3.43 |
| it.10 R2 | 16 | 32,768 | 9,600 | **93.60** |
| it.11 diagnosis | 8 | 2,048 | 2,400 | 0.43 |
| it.12-14 composed @R1 | 8 | 2,048 | 2,400 | 0.43 |
| it.12-14 composed @R2 | 8 | 32,768 | 2,400 | 11.70 |
| it.15-17 R3 | 64 | 4,096 | 2,400 | 7.85 |
| it.18-21 R5 | 16 | 8,192 | 2,400 | 4.48 |
| it.22-23 R6 | 32 | 512 / 2,048 | 2,400 | 1.02 |
| **TOTAL** | **168** | | | **122.94 h** |

Cuts 1–4 bring the CPU plan from 637.28 h to **122.94 h** — still 5.1 days, and
that is with R2's 93.60 h untouched, because R2 **cannot** be cut. It is the
deciding cell; it is the reason this node exists; and at `n = 32768` on CPU it
costs 45.72 h per arm at the full ladder and 0.71 h per arm at the 150-step
floor. If the round can afford exactly one expensive thing, it is R2.

After every cut, **R2 is 76% of what is left** (93.60 of 122.94 h). That is the
shape a correctly-costed plan should have: the deciding measurement is the
expense, and everything else has been trimmed to fit around it. On CUDA the same
trimmed plan costs **9.61 h**, and the untrimmed one 46.99 h — which is why §6.3
opens with a device move rather than a cut list.

**The one cut that is NOT recommended: dropping R2's `n` to 8192 to match the
last round.** That is the cell that already read `NO READING` at
`eval_nrmse = 1.0252` with the remaining rungs journalled unaffordable. Repeating
it at the same `n` buys the same verdict. The whole point of `n = 32768` is that
it is four times the size of the cell that failed, and §5 says it fits.

---

## 7. THE PROBE OUTPUT, VERBATIM

`python scripts/v15_neptune_sizing_probe.py`, 2026-08-31, repo at `089e789`.

```
==============================================================================
A. ENVIRONMENT
==============================================================================
torch 2.5.1+cu121   cuda_available=True
gpu            NVIDIA GeForce RTX 4060 Laptop GPU  sm_89  MPs=24
vram total     7.996 GiB (8585216000 B)
vram free      6.939 GiB (7451181056 B)
host RAM       15.71 GiB total, 6.49 GiB available (58.7% used)
cell shape     s=64 d_model=16 layers=1 heads=1, FULL BATCH (batch == n)

==============================================================================
B1. MEMORY MODEL vs ALLOCATOR -- n sweep at s=64, fp32, CUDA
==============================================================================
       n   pred_resid      pred_op   pred_total     measured     m/p
     256        0.018        0.015        0.033        0.039   1.191
     512        0.035        0.030        0.066        0.063   0.950
    1024        0.070        0.061        0.132        0.125   0.950
    2048        0.141        0.122        0.263        0.250   0.950
   (GiB; m/p = measured / predicted. Linear in n, so this ratio
    is the ratio at n = 32768 as well.)

==============================================================================
B2. TERM SEPARATION -- s sweep at n=256, fp32, CUDA
==============================================================================
     s   pred_total     measured meas_B/(n*s*s)     m/p
    64       0.0329       0.0313      32.0425   0.950
   128       0.0963       0.0713      18.2500   0.740
   256       0.3146       0.2578      16.5012   0.820
   512       1.1166       1.0156      16.2503   0.910

back-solved from s=64 and s=512 (exact, 2 eqns 2 unknowns):
   C_RESIDUAL  module 18.00   this shape  17.92
   C_OPERATOR  module  3.90   this shape   3.50

==============================================================================
C. WALL CLOCK -- CPU, threads=12, median of 12 timed steps
==============================================================================
       n         s/step      s/150 steps
    2048         0.0810             12.2
    4096         0.1792             26.9
    8192         0.4246             63.7
   16384         0.9922            148.8

log-log OLS over 4 points: slope 1.2088  R^2 0.999688
endpoint slope n=2048..16384:      1.2049
inherited (V13_DAG_TASKLIST.md:88-91): endpoint 1.33813, 3-pt OLS 1.31700
log residuals: +0.0179, -0.0261, -0.0014, +0.0096

==============================================================================
D. WALL CLOCK -- CUDA, median of 12 timed steps
==============================================================================
       n         s/step      s/150 steps        cpu/gpu
    2048         0.0110              1.7            7.3x
    4096         0.0201              3.0            8.9x
    8192         0.0403              6.0           10.5x
   16384         0.0804             12.1           12.3x

log-log OLS over 4 points: slope 0.9602  R^2 0.998844

==============================================================================
F. R2 AT FULL SIZE -- n=32768 MEASURED, not extrapolated
==============================================================================
predicted activations at n=32768, s=64, fp32: 4.216 GiB (residual 2.250 + operator 1.950 + head 0.0156)
plus x_train 32768*64*16*4 = 0.125 GiB resident
working set the model says R2 needs: 4.341 GiB
free VRAM 6.887 GiB -> ATTEMPTING
  CUDA measured peak 4.000 GiB   measured/predicted 0.949
  CUDA 0.1605 s/step  ->  150 steps x 8 seeds = 0.054 h;  9600-step ladder x 8 seeds = 3.42 h
host RAM available 5.293 GiB -> ATTEMPTING
  CPU  2.1432 s/step  ->  150 steps x 8 seeds = 0.714 h;  9600-step ladder x 8 seeds = 45.72 h

==============================================================================
G. LARGEST AFFORDABLE n ON THIS GPU
==============================================================================
measured bytes/example: activations 131086 + x_train 4096 = 135182
eval-set reserve held back: 0.200 GiB (x_eval at n_eval=4096 plus its no-grad forward)
    margin     budget GiB        largest n
        0%          6.687            53112
       10%          6.018            47800
       20%          5.349            42489
       30%          4.681            37178

==============================================================================
E. R1 / R2 PROJECTION from the fits above
==============================================================================
R1  BED-M t*=2 n=2048
     predicted activations  0.263 GiB (residual 0.141 + operator 0.122), fp32
     cpu  0.0796 s/step -> 150 steps x 8 seeds = 0.03 h
     cuda 0.0107 s/step -> 150 steps x 8 seeds = 0.00 h
R2  BED-M t*=8 n=32768
     predicted activations  4.216 GiB (residual 2.250 + operator 1.950), fp32
     cpu  2.2715 s/step -> 150 steps x 8 seeds = 0.76 h
     cuda 0.1538 s/step -> 150 steps x 8 seeds = 0.05 h

{
 "backsolved": {
  "C_RESIDUAL": 17.923209054129465,
  "C_OPERATOR": 3.498567853655134
 }
}
```

Sections B/C/D report the four-point fits the script computed inline; §4.2's
five-point fits add the `n = 32768` anchor measured in §F and are computed from
the same points (CPU slope 1.1920, R² 0.999642; CUDA slope 0.9734, R² 0.999316).

---

## 8. LIMITS

Everything this node could not establish, collected once.

- **The composed arm's `C_OPERATOR` is NOT MEASURED.** ARM PL does not exist
  until it.6–7. Every memory number here is for a single-operator arm
  (`bench._softmax_operator`). A composed S-M + S-K arm may retain twice as many
  `[n,s,s]` tensors; §2.5 prices that case at 5.74 GiB, which still fits but
  leaves 1.0 GiB. Re-run this probe against ARM PL before it.9 commits.
- **BED-K's `s` and `n` are `[ASSUMED]`, not contract-fixed.** §2.5. If the
  power-law bed needs `s = 256`, R4's matched-`n` requirement caps both beds at
  `n ≈ 5,273` and R2's `n = 32768` cannot enter the two-sided table.
- **The 1.5–2.0x absolute speed gap against the inherited timings is NOT
  DIAGNOSED.** Only the *slope* was needed for the projections and the slope was
  re-measured; the level shift is noted and not explained.
- **The bf16-autocast saving (4.216 → 3.109 GiB predicted) is NOT MEASURED.**
  It would also change the arm's numerics, which is a capability question and
  therefore L-LEAN territory for this node.
- **No arm was trained, fitted or scored.** All timings and allocations used
  `torch.randn` data; the arm and optimizer are the shipped objects so the shapes
  are real, but no NRMSE, no seed sweep, no verdict and no capability number was
  produced. Step cost is a function of tensor shape, not tensor values.
- **The 0.200 GiB eval reserve in §5 is `[ASSUMED]`,** a rounded-up bound rather
  than a measurement; removing it moves `n_max` by 1,589.
- **VRAM free is a snapshot.** 6.939 GiB was free with 1.057 GiB already held by
  the display. A graphical workload started mid-run reduces it. R2's 38% headroom
  is the reason that is a nuisance rather than a failure.
- **`house-events.jsonl` was not read.** It was not needed; no ledger query
  arose. It was never grepped.
