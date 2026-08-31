# V17-K G0.6 / G0.7 — K-CERT AND K-COST

The certificate half and the pricing half of Gate 0. `scripts/k_cert.py` →
`results/k_cert_local.json`, run once locally as its own regression test;
`scripts/k_cost.py` reads that certificate and prices the Kaggle train.
`tests/gate0/test_g06_kcert.py` pins the script's own invariants, each with a
planted negative. `COSTS.md` carries the two device certificate slots.

**Provenance tags.** `[MEASURED]` = read off this box in this run.
`[FITTED]` = projected from a law fitted here, with its R². `[MODULE]` =
computed by a named function. `[INHERITED]` = carried from an earlier file,
cited. `[ASSUMED]` = a value no measurement fixes, with the reason. Untagged
numbers are struck.

**L-TIME.** No timing figure below is inherited. `V16_DEVICE_CERT.md` §4.1's
CUDA law `exp(−11.9670) · n^0.9734` and its arm multipliers appear only as the
things this run is compared **against**.

**GPU window.** Device work started **2026-08-31 16:01:20 UTC** and finished
**2026-08-31 17:17:21 UTC**. The card read `0 MiB used, 0 % utilisation` at the
start. Two timings that looked contended were re-run rather than published —
§4 says which, and the re-runs changed the instrument, not only the number.

**Git.** `git rev-parse HEAD` at start and end: **`ab5b48547884e04258276e6e808d5a71ea65f917`**,
unchanged. `git status --porcelain` was **empty at start**; at end it carries
five `M` files and eighteen `??` entries written by the five other agents on this
tree. §2 reads the two that could have moved a number in this report rather than
asserting they did not. **No writing git command was run by this node.**

---

## 1. LEAD CAVEAT — what a reader who takes the headline and stops gets wrong

The headline is *"G0.6 GREEN, G0.7 GREEN; both laws refit at R² ≥ 0.99999; the
worst bar δ/tol is 8.58 % against a 50 % line."* Four things that headline hides,
and the first two decide whether the round can run.

1. **The workhorse arm's law is fitted over three points, and the round's
   deciding cell is four doublings past the last of them.** `arm_smprime`'s law
   spans `n = 2048…8192` only, because at `n = 16384` and `n = 32768` the arm's
   caching allocator reserves **10.578 GiB and 13.969 GiB from a 7.996 GiB
   card** and the driver pages over PCIe — those two step costs are bus
   measurements, not arm measurements, and admitting them would have bent the
   exponent from 1.0026 to 1.2341 and dropped R² to 0.976933. So the two largest
   rows of the chunk table are **extrapolations**, and the largest of them says
   **R2's registered `n = 32768` does not fit a free Kaggle card either** —
   15.771 GiB projected against a 13.50 GiB T4 budget, 117 %.

2. **"Determinism at reduction length 64" is not one answer and the useful half
   is negative.** With the strict flag OFF, the arm's hop, forward and gradient
   are all bitwise over 8 repeats. With the flag ON, the hop and the forward are
   bitwise and **the gradient does not run at all**: `cumprod`'s backward is
   computed with `cumsum`, which has no deterministic CUDA kernel in torch
   2.5.1. Moving the arm from a prefix scan to a path product **did not escape
   the missing kernel — it moved it from the forward to the backward.** Anyone
   who reads "bitwise at 64" and starts a training run under
   `use_deterministic_algorithms(True)` gets a `RuntimeError` at step 0.

3. **The memory model reproduced on three constants of four, and the fourth is
   wrong in the optimistic direction.** `C_RESIDUAL`, `C_OPERATOR` and the bf16
   operator bytes/element all confirm to within 2 %. `DTYPE_MODES
   ["bf16_autocast"][0]` reads **2.383 measured against 2.2 in the module,
   +8.3 %** — the module under-predicts residual activation memory. It is
   bounded: at the Q3 chunk shape it moves the prediction 2.715 → 2.738 GiB and
   `max_batch` is 45 either way, so no decision this round makes moves. It is
   still wrong and `ceq/sizing.py` is not this node's to edit.

4. **The failure ledger's good column prices an interface that is not
   committed.** `save_every`, `latest_checkpoint` and the two alternating slots
   exist in the **working tree** (`M ceq/hf/train.py`, another node's work in
   flight). At `git HEAD` they do not exist and a mid-chunk kill costs the whole
   chunk. Both columns are priced; the difference is 659 GPU-minutes per kill.

One more, smaller: the bar's four rungs, twenty clause readings and worst δ/tol
of 8.580 × 10⁻² **reproduce `V16_BAR_RECERT.md` §4 to every printed digit**.
That is a measured confirmation across a round boundary, not an inheritance —
but it also means the bar half of this certificate found nothing new, and a
reader should not credit it as independent evidence of anything except that the
bar has not moved.

---

## 2. VERDICT TABLE

| item | verdict | why |
|---|---|---|
| **G0.6 K-CERT** | **GREEN** | Script packaged, runs unchanged on either box, emits one machine-readable certificate. Throughput refit printed with R² (two laws, 0.999998 and 1.000000). Memory law refit against the allocator, R² 0.996373 / 0.999830. Bar re-certification: 20 rows, worst δ/tol **8.58 %**, clears the 50 % line by 5.83×, no HALT. 0-step gate: **30 / 30** cell shapes reachable and passing. Determinism at 64 reported in **both** regimes. Run once locally; `tests/gate0/test_g06_kcert.py` **13 passed**. |
| **G0.7 K-COST** | **GREEN** | Chunk table derived from *this* run's law at every candidate shape, with the extrapolated rows marked. Margin derived, not chosen (§8). Weekly quota arithmetic charges startup. Failure ledger: four kill points, each with its recovery path and its cost in GPU-minutes, priced against both the working-tree and the `git HEAD` checkpoint interface. |
| — sub-finding: **R2 at `n = 32768` for the workhorse arm** | **RED** | Does not fit the certified local card (13.969 GiB reserved on 7.996 GiB) and does not fit a T4-16GB either (15.771 GiB projected against a 13.50 GiB budget, 117 %). The last power of two that fits is **`n = 16384` at 59 % of the T4 budget**. |
| — sub-finding: **strict determinism during training** | **BLOCKED** | `use_deterministic_algorithms(True)` makes the workhorse arm's **backward** unexecutable on CUDA. Forward-only and replay are bitwise. This is the measured input to `V17K_RULINGS.md` RULING 1 and it agrees with that ruling's shape: strict for forward, `warn_only` for training. |
| — sub-finding: **`DTYPE_MODES["bf16_autocast"][0]`** | **RED, reported not edited** | 2.383 measured against 2.2 in the module. Correct value and its measurement in §5.2. |

### 2.1 What changed under this node while it ran

`git status --porcelain` was empty at start. Two of the five files that became
`M` could have moved a number here, and both were **read rather than assumed**:

- **`ceq/arm_smprime.py`** — the executable diff is two lines, both inside
  `label_cell`, splitting a `residual` recording into `residual_bos` and
  `residual`. Nothing on `magnitude`, `blend`, `gate`, `path_product`, `hop`,
  `operator` or `ArmSMPrime.forward` — the whole path every timing, allocation,
  gate reading and determinism reading below runs through. **Verified by
  read-only diff.**
- **`ceq/hf/train.py`** — gained `save_every`, `_save_checkpoint`,
  `_atomic_torch_save`, `latest_checkpoint` and two alternating slots. This
  landed *during* the run and it is why §10 prices two columns instead of one.
  Not edited by this node; read only.

---

## 3. RED EVIDENCE

Tests were written before the script existed. Verbatim:

```
$ python -m pytest tests/gate0/test_g06_kcert.py -q
=================================== ERRORS ====================================
_______________ ERROR collecting tests/gate0/test_g06_kcert.py ________________
ImportError while importing test module '...\tests\gate0\test_g06_kcert.py'.
tests\gate0\test_g06_kcert.py:29: in <module>
    from scripts import k_cert
E   ImportError: cannot import name 'k_cert' from 'scripts' (unknown location)
=========================== short test summary info ===========================
ERROR tests/gate0/test_g06_kcert.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.28s
```

With the pure helpers written but before the certificate existed — the eight
predicate tests pass and the five certificate-reading tests are RED, which is
the split the file is built to have:

```
$ python -m pytest tests/gate0/test_g06_kcert.py -q
E           Failed: results/k_cert_local.json missing -- run `python scripts/k_cert.py` once
=========================== short test summary info ===========================
FAILED ...::test_certificate_carries_both_refits_and_prints_an_r2_for_each
FAILED ...::test_certificate_worst_delta_over_tol_clears_the_fifty_percent_line
FAILED ...::test_certificate_zero_step_gate_fired_at_every_cell_shape_it_could_reach
FAILED ...::test_certificate_reports_determinism_in_both_regimes
FAILED ...::test_certificate_names_the_box_and_the_commit_so_kaggle_can_append
5 failed, 8 passed in 1.39s
```

GREEN after the run of record: **`13 passed in 0.85s`**.

### 3.1 The planted negatives — this repo has struck 14 vacuous controls

Every PASS half carries a FAIL half built from a fabricated reading. Three were
named in the item; two more are here because the same defect was available.

| planted negative | what it proves can fire | test |
|---|---|---|
| **a fabricated slow device** — the same exponent with 10× the intercept | the chunk table is a *function of the refitted law*: `steps_in_hours` returns exactly one tenth as many steps at every `n`. A constant wearing a fit's clothes would not move. | `test_a_fabricated_slow_device_moves_the_chunk_table_by_exactly_its_factor` |
| **a δ/tol at 60 %** — `delta = 6e-7` against `tol = 1e-6` | `halt_on_bar` returns a HALT string. Without it, `MIN`/`HALT_FRACTION` could be anything and the table would still read GREEN. | `test_a_bar_row_at_sixty_percent_of_tolerance_halts` |
| **a non-deterministic reduction** — deltas `[0, 1e-7, 0]` | `determinism_verdict` reads `bitwise False` and reports `max_abs = 1e-7`. The column can distinguish the two regimes it reports. | `test_a_fabricated_non_deterministic_reduction_reads_not_bitwise` |
| **a series with no power-law structure** — `[1.0, 5.0, 1.2, 9.0, 1.1]` | `refuse_weak_fit` raises `NotALaw`. An R² that never rejects anything is not a test, and a weak fit is **not filed under `laws`** — it goes to `rejected_laws` with its reason. This fired for real: see §4.2. | `test_a_series_with_no_power_law_structure_is_refused_on_r2` |
| **an arm that really beats the mean** — `gate(1.0, 0.95)`, and `gate(nan, 1.0)` | the 0-step gate rejects, and NaN is checked *first* because `float('nan') >= 1.0` is False and would pass a broken instrument silently. | `test_the_zero_step_gate_admits_the_measured_null_tail_and_rejects_a_real_beat` |

The certificate-reading tests carry their own non-degeneracy check: the 0-step
test asserts `len({round(nrmse0_eval, 9)}) > 1` over the reachable cells, so a
gate that passed because every reading was the same constant would fail.

---

## 4. REFIT 1 — THE THROUGHPUT LAW

Median over **3 independent child processes** of the median of **≥12 timed steps
and ≥3.0 s of timed work**, 2 warm-up steps discarded. `s = 64`, `d_model = 16`,
`torch.randn` data, forward + backward + `Adam.step`, `threads = 8`.

### 4.1 The measurement `[MEASURED]`

| arm | n | s/step | ×150 | reserved GiB | children | cross-child spread | admitted |
|---|---|---|---|---|---|---|---|
| `softmax` | 2,048 | 0.010142 | 1.5 s | 0.416 | 3/3 | 0.13 % | yes |
| `softmax` | 4,096 | 0.020291 | 3.0 s | 0.770 | 3/3 | 0.26 % | yes |
| `softmax` | 8,192 | 0.040525 | 6.1 s | 1.502 | 3/3 | 0.09 % | yes |
| `softmax` | 16,384 | 0.080681 | 12.1 s | 2.971 | 2/3 | 0.03 % | yes |
| `softmax` | 32,768 | 0.160665 | 24.1 s | 5.908 | 3/3 | 0.01 % | yes |
| `arm_smprime` | 2,048 | 0.093052 | 14.0 s | 1.287 | 3/3 | 0.19 % | yes |
| `arm_smprime` | 4,096 | 0.186518 | 28.0 s | 2.297 | 3/3 | 0.17 % | yes |
| `arm_smprime` | 8,192 | 0.373543 | 56.0 s | 5.711 | 3/3 | 0.02 % | yes |
| `arm_smprime` | 16,384 | 1.189290 | 178.4 s | **10.578** | 2/3 | 1.36 % | **NO — reserved > card** |
| `arm_smprime` | 32,768 | 15.174861 | 2276.2 s | **13.969** | 3/3 | 1.36 % | **NO — reserved > card** |

**Fitted laws, log-log OLS, R² on the log residuals** `[FITTED]`:

| arm | law | exponent | R² | fitted over |
|---|---|---|---|---|
| `softmax` | `s/step = exp(−12.1852) · n^0.9963` | **0.9963** | **0.999998** | 2048…32768, 5 points |
| **`arm_smprime`** | `s/step = exp(−10.0187) · n^1.0026` | **1.0026** | **1.000000** | 2048, 4096, 8192, 3 points |

**Against what would otherwise have been carried.** `V16_DEVICE_CERT.md` §4.1
recorded CUDA `n^0.9734`, R² 0.999384 `[INHERITED, cited only]`. This round
reads **0.9963**, R² 0.999998 — 2.4 % higher and an order of magnitude tighter.
The `×150` column is identical to V16's at `n = 16384` (12.1 s) and `n = 32768`
(24.1 s) and differs at the small end, which is exactly where §4.3's instrument
change bites. **The V16 arm multipliers are not carried:** `arm_smprime` is
fitted directly, because a multiplier measured at one shape is a claim that the
ratio is constant in `n`, and this arm's exponent (1.0026) differs from
softmax's (0.9963) by more than either fit's residual.

### 4.2 The R² gate fired for real, twice

Not a hypothetical. On the run where `n = 16384` was admitted, `arm_smprime`
fitted `exp(−11.8909) · n^1.2341` at **R² 0.976933**, and `refuse_weak_fit` kept
it out of `laws` and put it in `rejected_laws`. On another, only two points were
admitted and the fit was refused for having fewer than three. **The gate is the
reason the published exponent is 1.0026 and not 1.2341**, and 1.2341 would have
mispriced an 11 h chunk at `n = 8192` by 1.6×.

### 4.3 The two timings that looked contended, and what re-running them changed

The item requires re-running anything that looks contended. Two did, and neither
re-run was a repeat of the same measurement.

- **`arm_smprime` at `n = 16384`** died with `RuntimeError: CUDA error: out of
  memory` in the first pass and completed in the second. Run three times
  standalone: **1 hard OOM, 2 completions at 1.4273 and 1.3811 s/step, with the
  in-run spread 0.77–1.27 s — 55–92 % of the median.** That is not a step cost;
  it is an eviction. §4.4 says how the script now detects it without a threshold.
- **Every shape at small `n`.** Two full passes of the original 12-step window
  read `softmax @ 2048` at 0.010200 and 0.011836 s/step — **14 % apart** — and
  moved `arm_smprime`'s exponent between 0.8698 and 1.0034 and its R² between
  0.993309 and 0.999996. Twelve steps at `n = 2048` is **0.12 s of work** and a
  laptop GPU is still ramping its clock inside it. The window is now bounded in
  **both** steps and seconds (`MIN_TIMED_SECONDS = 3.0`), and the cross-child
  spread column above — 0.01 % to 0.26 % on every admitted shape — is what that
  bought.

### 4.4 Two admission rules, both measurements, neither a chosen threshold

- **Residency, on RESERVED bytes and not allocated.** At `n = 16384` the arm
  **allocates 6.343 GiB** — comfortably under the 7.996 GiB card — while
  **reserving 11.721 GiB** from the driver, and runs at 1.172 s/step against the
  0.747 s/step the in-VRAM law extrapolates, **3.1×**. At `n = 8192` it
  allocates 3.179 GiB and reserves 5.289 GiB, 0.66 of the card, and runs at
  0.377 s/step, on the law. **A test on allocated peak calls `n = 16384`
  resident and puts a PCIe number in the law.** The test is
  `max_memory_reserved > total_memory`: the allocator asked the card for more
  than it has.
- **A majority of children must complete.** Distinct from residency, because it
  is a different failure. `arm_smprime @ n = 4096` — 1.9 GiB reserved on an
  8 GiB card, resident by a factor of four — lost one child of three to a hard
  Windows process abort (exit `0xC0000409`) in one pass and completed 3/3 in
  the next. Excluding a demonstrably resident shape because the driver fell over
  once deletes a real point and bends the exponent, so deaths are **recorded**
  and residency does the excluding.

### 4.5 One process per shape, and why that is not over-engineering

An exhausted 8 GiB card raises `RuntimeError: CUDA error: out of memory` — the
**driver's** OOM, not the caching allocator's `torch.cuda.OutOfMemoryError` —
and that error **poisons the context**: the very next `torch.cuda.empty_cache()`
raises the same thing, and every later shape in the same process reads NO FIT
whether or not it fits. Measured, on the first run of this script. An in-process
`try/except` can therefore only find the *first* shape past the boundary, never
the boundary. One child per shape is the smallest thing that makes "does this
cell fit" a measurement, and a child that dies for any reason is recorded as not
fitting with its reason, so a crash can never read as a pass.

---

## 5. REFIT 2 — THE MEMORY LAW, AGAINST THE CUDA ALLOCATOR

### 5.1 The re-solve

`ceq/lm.py::TinyLM`, **B = 4, d = 256, L = 4, H = 4**, forward + backward, peak
allocator bytes above the resident baseline — the shape `ceq/sizing.py`'s own
docstring names, so agreement is a reproduction and not a nearby claim. The two
terms are separated by the **seq sweep** and not by assertion: the residual term
grows as `L·B·S·d` and the operator term as `L·H·B·S·S`, which are linearly
independent in `S`. Two-parameter least squares; the head term is held at the
module's structural `C_HEAD = 2` and subtracted.

| constant | `ceq/sizing.py` | re-solved `[MEASURED]` | R² | Δ | verdict |
|---|---|---|---|---|---|
| `C_RESIDUAL`, fp32 (4.0 B/elem) | 18 | **17.874** | 0.996373 (8 points) | −0.70 % | **CONFIRMED** |
| `C_OPERATOR`, fp32 (4.0 B/elem) | 3.9 | **3.823** | 0.996373 (8 points) | −1.97 % | **CONFIRMED** |
| bf16 autocast, **operator** B/elem | 3.4 | **3.341** | 0.999830 (6 points) | −1.74 % | **CONFIRMED** |
| bf16 autocast, **residual** B/elem | 2.2 | **2.383** | 0.999830 (6 points) | **+8.32 %** | **WRONG, optimistic** |

The fp32 row is also a check on itself: dividing the fitted residual coefficient
by the module's count of 18 gives **3.972 effective bytes per element** against
the 4.0 the dtype actually is — 0.7 % — so the count and the width are
independently consistent rather than trading off against each other.

### 5.2 The one constant that does not reproduce, and exactly what it costs

`DTYPE_MODES["bf16_autocast"] = (2.2, 3.4)`. The measured pair on this box in
this run is **(2.383, 3.341)**. The operator term confirms; the residual term is
**8.3 % low**, which is the direction that under-predicts memory.

Priced at the shape the round actually runs — Q3, d 512 / L 8 / H 8 / seq 512 /
batch 8, `T4-16GB`:

| | module (2.2, 3.4) | measured (2.383, 3.341) |
|---|---|---|
| `signed` total | 2.715 GiB | **2.738 GiB** (+0.85 %) |
| `softmax` total | 1.058 GiB | 1.109 GiB |
| `max_batch(signed)` | 45 | **45** |

**No decision this round makes moves.** The operator term dominates at this
shape, and the residual error is diluted. It matters where the residual term
dominates — many layers, short sequence — which this round does not visit.
**`ceq/sizing.py` is not edited by this node**; the correct value is recorded
here and in `COSTS.md` §1.2 with its measurement.

### 5.3 Residency, per cell shape `[MEASURED]`

| arm | n | allocated GiB | reserved GiB | operator dtype | resident on 7.996 GiB |
|---|---|---|---|---|---|
| `softmax` | 2,048 → 32,768 | 0.282 → 4.032 | 0.354 → 4.908 | `float32` @ 4 B | yes, all five |
| `arm_smprime` | 2,048 | 0.814 | 0.988 | `complex64` @ 8 B | yes |
| `arm_smprime` | 4,096 | 1.597 | 1.908 | `complex64` @ 8 B | yes |
| `arm_smprime` | 8,192 | 3.163 | 3.777 | `complex64` @ 8 B | yes |
| `arm_smprime` | 16,384 | 6.296 | 7.504 / **10.578 under a training loop** | `complex64` @ 8 B | **NO in training** |
| `arm_smprime` | 32,768 | 12.559 | 13.951 | `complex64` @ 8 B | **NO** |

The `n = 16384` row carries two reserved figures and the difference is the
finding: a bare forward+backward reserves 7.504 GiB and stays (just) resident; a
training loop, holding Adam's state and churning allocations, reserves
10.578 GiB and pages. **The operative number for pricing a chunk is the training
loop's**, because that is what a chunk is.

The operator's **8 bytes per element** is read off the live tensor, not assumed:
`ArmSMPrime` forms a `complex64` `[n, s, s]` when its parameters are fp32, which
is 2× the softmax arm's `float32` at the same shape and is the whole reason this
arm's residency boundary sits two doublings below softmax's.

---

## 6. THE BAR RE-CERTIFICATION — δ/tol

`calibrate_bar(n = N_EVAL = 4096, s = 64, d = 24, steps = 600, lr = 0.02,
seed = 0)` through the registered `e3_t*` hooks — the exact call
`r10_capacity_sweep.main()` makes. CPU pass at the shipped defaults
(`deterministic_algorithms False`); CUDA pass under
`use_deterministic_algorithms(True)`, `cudnn.deterministic = True`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, exactly as `main()` sets them.

**Every tolerance is quoted, not chosen** — from `bar_verdict`'s body or
`flipper_tol`'s default, text that predates this node:

| clause | tolerance | source |
|---|---|---|
| `predict_the_mean` | `abs(x − 1.0) > 1e-6` ⇒ FAIL | `bar_verdict` body |
| `payload_only` | one-sided `x >= 1.0`; δ is read against the reading's own margin `x − 1.0` | `bar_verdict` body |
| `oracle` | `x < 1e-6` | `bar_verdict` body |
| `flipper_dependence` | `abs(x − closed form) > 0.05` ⇒ FAIL | `bar_verdict(flipper_tol=0.05)` default |
| `trained_two_feature` | one-sided `x < 1.0`; δ against the margin `1.0 − x` | `bar_verdict` body |

Verdict on all four rungs, both devices: **BAR CALIBRATED**.

### 6.1 The table, worst first `[MEASURED]`

| task | clause | CPU | CUDA | δ | tol / margin | **δ/tol** |
|---|---|---|---|---|---|---|
| **`e3_t8`** | **`predict_the_mean`** | 1 | 1.0000000858002478 | 8.580e-08 | 1.000e-06 | **8.580e-02** |
| `e3_t2` | `predict_the_mean` | 1 | 1.0000000843170462 | 8.432e-08 | 1.000e-06 | 8.432e-02 |
| `e3_t2` | `flipper_dependence` | 1.4012436552018552 | 1.4012437605053387 | 1.053e-07 | 5.000e-02 | 2.106e-06 |
| `e3_t8` | `payload_only` | 1.0640782793345231 | 1.0640783706327031 | 9.130e-08 | 6.408e-02 *(margin)* | 1.425e-06 |
| `e3_t32` | `flipper_dependence` | 0.368704564751893 | 0.36870459153009694 | 2.678e-08 | 5.000e-02 | 5.356e-07 |
| `e3_t1` | `payload_only` | 1.4120774691675735 | 1.4120773501636601 | 1.190e-07 | 4.121e-01 *(margin)* | 2.888e-07 |
| `e3_t2` | `trained_two_feature` | 0.013981593578261986 | 0.013981630466969725 | 3.689e-08 | 9.860e-01 *(margin)* | 3.741e-08 |
| `e3_t8` | `trained_two_feature` | 0.022470289307256582 | 0.022470303300872813 | 1.399e-08 | 9.775e-01 *(margin)* | 1.432e-08 |
| `e3_t32` | `trained_two_feature` | 0.024553869773872392 | 0.024553861762042058 | 8.012e-09 | 9.754e-01 *(margin)* | 8.214e-09 |
| `e3_t1` | `trained_two_feature` | 0.0082265731871092578 | 0.0082265787654177015 | 5.578e-09 | 9.918e-01 *(margin)* | 5.625e-09 |
| *(remaining 10 rows)* | `oracle` ×4, `predict_the_mean` ×2, `payload_only` ×2, `flipper_dependence` ×2 | | | **0 exactly** | | **0** |

### 6.2 The worst row against the 50 % line

> **WORST δ/tol = 8.580 × 10⁻² = 8.58 %, at `e3_t8` / `predict_the_mean`.**
> The line is **50 %**. **Headroom 5.83×. NO HALT.**

Half the twenty readings are **exactly zero** — bit-identical across the device.
The two rows that carry any weight at all are both `predict_the_mean`, and that
clause is an **algebraic identity** (`nrmse(mean, y) ≡ 1`) evaluated in float32:
it is fp32 accumulation over `n = 4096` examples, not a device effect.
`V16_BAR_RECERT.md` §4 measured `abs(cpu − 1.0)` reaching the same order as
`abs(cuda − 1.0)`, and the same reading here — worst δ/tol `8.580025e-02` — is
**identical to V16's to every printed digit**, across a round boundary and a
separate script.

**HALT is reachable.** `halt_on_bar` is exercised on a fabricated 60 % row in
`test_a_bar_row_at_sixty_percent_of_tolerance_halts`, and returns the HALT
string; the script prints it and exits 2. It is not decoration.

---

## 7. THE 0-STEP RED GATE, AND DETERMINISM AT 64

### 7.1 The gate at every cell shape `[MEASURED]`

`train_with_checkpoints`'s gate, unchanged: NaN checked **first** — because
`float('nan') >= 1.0` is False and would pass a broken instrument silently —
then `r0t >= 1.0 − GATE_TOL` and `r0e >= 1.0 − GATE_TOL`, `GATE_TOL = 1e-3`.
Arm constructed then moved, so the init bytes are the ones every cpu reading in
`results/` was taken through. Eval split drawn once at `N_EVAL = 4096`, seed
12345, task `e3_t2`.

**2 arms × 5 values of `n_train` × 3 seeds = 30 cell shapes. 30 reachable.
30 pass.**

| | `n_train` = 2,048 | 4,096 | 8,192 | 16,384 | 32,768 |
|---|---|---|---|---|---|
| `softmax`, seeds 0/1/2 | OK OK OK | OK OK OK | OK OK OK | OK OK OK | OK OK OK |
| `arm_smprime`, seeds 0/1/2 | OK OK OK | OK OK OK | OK OK OK | OK OK OK | OK OK OK |

Worst margin above the `1.0 − GATE_TOL` line over all 30: **1.9572 × 10⁻³**,
i.e. **1.96× the tolerance**. Readings span 1.000957 to 1.018091, so the gate
was evaluated on thirty distinct numbers and not on a constant — asserted by the
test.

The gate runs under `no_grad`, which is why it is reachable at every shape
including the two that page: a forward at `n = 32768` fits where a training step
does not.

### 7.2 Determinism at the arms' reduction length, in **both** regimes `[MEASURED]`

`s = 64` is the arms' reduction length in both contractions the operator
performs — the masked cumulative product along a row, and the `a @ x` sum over
`j`. Workhorse arm `ceq/arm_smprime.py`, 8 repeats of each quantity on identical
inputs, max absolute deviation from the first call.

| quantity | flag OFF | flag ON |
|---|---|---|
| `hop` — masked `cumprod` over 64 | **bitwise**, max\|Δ\| **0.0** | **bitwise**, max\|Δ\| **0.0** |
| `ArmSMPrime.forward` — hop + `a @ x` over 64 | **bitwise**, max\|Δ\| **0.0** | **bitwise**, max\|Δ\| **0.0** |
| **gradient** — backward of both | **bitwise**, max\|Δ\| **0.0** | **NOT EXECUTABLE** |

```
RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation,
but you set 'torch.use_deterministic_algorithms(True)'.
```

**The finding, stated precisely.** `ceq/arm_smprime.py` reduces with **`cumprod`,
not `cumsum`** — `path_product` is a masked cumulative product and the module
docstring is explicit that no prefix scan represents the hop. `torch.cumprod`'s
**forward** is executable under the strict flag on CUDA in torch 2.5.1.
**Autograd differentiates a cumulative product with `cumsum`**, so the
**backward** hits the same missing kernel that blocks `arm_phase` and `arm_pl`.
Isolated directly:

| call | flag ON |
|---|---|
| `torch.cumprod(m, -1)` forward | **OK** |
| `torch.cumprod(m, -1)` forward + backward | **RAISES** `cumsum_cuda_kernel …` |
| `arm_smprime.hop(m, th)` (no grad) | **OK** |
| `ArmSMPrime.forward` under `no_grad` | **OK** |
| `ArmSMPrime` forward + backward | **RAISES** `cumsum_cuda_kernel …` |
| `Arm("softmax")` forward + backward | **OK** |
| bare `torch.cumsum` forward + backward (what `arm_phase`/`arm_pl` reduce with) | **RAISES** |

**Neither regime is picked as "the" answer here.** What the certificate states
is the split: **bitwise for every forward-only and replay quantity in both
regimes; the strict flag is unexecutable for training on this arm.** That is
also the measured input to `V17K_RULINGS.md` **RULING 1**, and it agrees with
that ruling's shape — strict mode ON for deciding inference cells, the hole
inherited between checkpoints — while adding the part the ruling did not have:
**the hole is specifically in `cumprod`'s backward, and it is not escaped by the
choice of path product over prefix scan.**

`calibrate_bar` on CUDA *is* executable under the flag (§6 ran it that way), and
so is the 0-step gate. **The certified deterministic path covers the bar and the
gate; it does not cover a gradient step.**

---

## 8. THE CHUNK TABLE — G0.7

### 8.1 The margin: why 11 h and not 12

The margin is **derived from measured components plus two named assumptions**,
not chosen.

| component | seconds | tag |
|---|---|---|
| `import torch` + `transformers`, timed in a child | 2.74 | `[MEASURED]` |
| `ceq.hf.train.preflight` at the Q3 shape | 0.000 | `[MEASURED]` |
| final checkpoint write through the atomic path | 0.609 | `[MEASURED]` |
| one ragged step at the Q3 shape | 0.299 | `[MEASURED]` |
| **measured floor** | **3.65** | |
| pip install of the repo's own requirements | 180 | `[ASSUMED]` — a Kaggle image already carries torch, so this is the repo's extras only; this box cannot see Kaggle's network |
| dataset attach + corpus build | 300 | `[ASSUMED]` — G0.5 owns the real number; 300 s is the order the notebook's data cells imply |
| **total charged** | **484** | |
| **margin available** = (12 − 11) h | **3600** | |
| **covered** | **7.4×** | |

`TRAINING.md` §4 prescribes `0.80 × session_cap` = 9.6 h `[INHERITED]`. The
contract's "≤11 h" is *less* conservative than that rule, and the arithmetic
above is why it is still safe: the fixed cost a chunk actually pays is 484 s,
and 3600 s covers it 7.4× even with both assumptions inflated to their stated
orders. **If G0.5 measures the corpus build above ~55 minutes, this margin
fails and the chunk must shorten** — that is the one input that could move it.

### 8.2 The table — steps in ≤ 11 h at every candidate shape `[FITTED]`

`s/step` from §4.1's refitted laws. `alloc` is the measured allocated peak;
`proj resv` applies the **worst reserved/allocated ratio measured over this
box's resident shapes, 1.256** `[MEASURED]`, to it. The budget is
`ceq/sizing.py::GPUS["T4-16GB"]` = 15.0 GiB × 0.90 = **13.50 GiB** `[MODULE]`.

| arm | n | s/step | **steps / 11 h** | alloc GiB | proj resv GiB | T4-16GB |
|---|---|---|---|---|---|---|
| `arm_smprime` | 2,048 | 0.093066 | **425,506** | 0.814 | 1.023 | yes (8 %) |
| `arm_smprime` | 4,096 | 0.186465 | **212,372** | 1.597 | 2.006 | yes (15 %) |
| `arm_smprime` | 8,192 | 0.373597 | **105,996** | 3.163 | 3.973 | yes (29 %) |
| `arm_smprime` | 16,384 | 0.748531 | **52,903** | 6.296 | 7.906 | yes (59 %) — **extrapolated** |
| `arm_smprime` | 32,768 | 1.499743 | **26,404** | 12.559 | 15.771 | **NO (117 %)** — **extrapolated** |
| `softmax` | 2,048 | 0.010162 | 3,896,952 | 0.282 | 0.354 | yes (3 %) |
| `softmax` | 4,096 | 0.020271 | 1,953,546 | 0.532 | 0.668 | yes (5 %) |
| `softmax` | 8,192 | 0.040436 | 979,314 | 1.032 | 1.295 | yes (10 %) |
| `softmax` | 16,384 | 0.080663 | 490,931 | 2.032 | 2.551 | yes (19 %) |
| `softmax` | 32,768 | 0.160907 | 246,104 | 4.032 | 5.063 | yes (38 %) |
| **LM (Q3)** d512 L8 H8 seq512 b8 | — | 0.299025 | **132,430** | 3.208 | 4.029 | yes — `[MEASURED]` directly |

The Q3 row is measured and not fitted, because §4's ladder is the research-cell
ladder and the Q3 chunk is a different shape: 25,728,000 parameters,
`grad_finite True` `[MEASURED]` — the only thing read off a gradient in this
node, per L-LEAN. One 11 h chunk is **132,430 steps = 542,433,280 tokens**.

**The two extrapolated rows are labelled at the point of use** and the largest
of them is the round's problem: **`arm_smprime` at `n = 32,768` — R2's
registered `n` — does not fit a free Kaggle card.** The last power of two that
does is **`n = 16,384`, at 59 % of the T4 budget**. This is the same shape of
finding `V16_DEVICE_CERT.md` §3.4 recorded for `arm_phase` at the same `n`,
arrived at independently for a different arm on a different card.

**A caveat on the T4 column.** Reserved bytes are an allocator behaviour and
depend on fragmentation, driver and card. The projection is honest about being
a projection: it uses the measured *allocated* peak, which is a physical
requirement of the shape and device-independent, and inflates it by a ratio
measured only in the resident regime. **L2 must re-read residency on the Kaggle
card rather than trusting this column**, and `COSTS.md` §2 has the slot. The
1.256 ratio is itself the conservative end: it comes from `softmax @ n = 2048`,
the *smallest* resident shape, where the allocator's fixed block overhead is
proportionally largest; the three largest resident shapes read 1.217–1.218.

---

## 9. THE WEEKLY QUOTA

`[INHERITED, TRAINING.md:142 citing README.md:537-544]`, not re-measurable from
this box: **12 h session cap, 30 GPU-h/week, 20 GB `/kaggle/working`.**

| | value |
|---|---|
| naive `30 / 11` | 2.73 chunks — **wrong** |
| billed per chunk | 11.000 h training **+ 0.134 h startup** = **11.134 h** |
| whole chunks per week | **2** |
| remainder | **7.731 h**, of which **7.597 h** is trainable |
| total trainable per week | 2 × 11.000 + 7.597 = **29.597 h** |
| at the Q3 shape | **356,321 steps = 1,459,490,816 tokens per week** |

**Why the naive figure is wrong.** Kaggle bills the *session*, not the
optimiser. Install, import, dataset attach, corpus build and the final save all
burn quota. Counting `30 / 11 = 2.73` charges nothing for three startups. The
remainder is still a usable chunk — 7.597 h of training — but it is **not** a
third 11 h chunk, and a plan that assumes 2.73 × 11 h = 30 h of *training* is
over-budget by 0.4 h before anything goes wrong.

---

## 10. THE FAILURE LEDGER

`save_every` is **solved, not chosen**: `k` checkpoints cost `k × 0.609 s`
against an 11 h chunk, so a ≤1 % wall-clock overhead budget gives
**`save_every = 204` steps, 650 writes per chunk, 395.2 s = 0.998 %.**

| overhead budget | `save_every` | writes/chunk | worst lost GPU-min | bytes written over the chunk |
|---|---|---|---|---|
| 0.1 % | 2,038 | 65 | 10.16 | 20.1 GB |
| 0.5 % | 408 | 325 | 2.03 | 100.4 GB |
| **1.0 %** | **204** | **650** | **1.02** | **200.8 GB** |
| 5.0 % | 41 | 3,251 | 0.20 | 1,004.2 GB |

Those byte figures are **write volume, not capacity** — the two slots are
overwritten in place, so occupancy stays at the 6.18 % four-directory steady
state `[INHERITED, V17_G02_G03_CHECKPOINT.md §8]`. 200.8 GB over 11 h is 5 MB/s
sustained. The choice of row is the author's; 1 % is priced below.

### 10.1 The four kill points

Priced twice. The **left** column is `ceq/hf/train.py` **as it stands in the
working tree** — `save_every`, `_save_checkpoint`, `_atomic_torch_save`,
`latest_checkpoint`, two alternating slots, landed by another node *during* this
run and **not committed**. The **right** column is `git HEAD`, which has a
single end-of-loop save and nothing else.

| kill point | recovery path | with periodic (worst lost) | without (worst lost) |
|---|---|---|---|
| **during warm-up** — install / import / dataset attach / corpus build, before step 0 | restart the chunk from `latest_checkpoint(prev_out_dir)`; the previous chunk's directory was never opened for writing and is untouched by construction. No gradient step has run, so nothing trainable is lost either way. | **8.06 GPU-min** | 8.06 GPU-min |
| **mid-chunk** — session cap, kernel death, quota exhaustion | `latest_checkpoint(out_dir)` returns the newer of the two slots; resume into a **new** `out_dir` (`train()` now refuses `resume_from == out_dir`). Expected loss is half a period under a uniform kill time; worst is one full period. | **1.02 GPU-min** | **660.00 GPU-min** — the whole elapsed chunk |
| **during a checkpoint write** — `save_pretrained` shards, then the atomic `trainer_state.pt` | *With slots:* the slot being written is the one that dies; the **other** slot is complete and untouched, and `latest_checkpoint` skips the half-written one because its `trainer_state.pt` does not load. Alternating is what makes this recoverable — a single slot overwritten in place would be destroyed at exactly the moment it is being replaced. *Without:* there is one directory and it is the one being written; a kill inside `save_pretrained` leaves partial shards beside a **stale** `trainer_state.pt`, which **loads** — so the directory reads resumable and is not. **Strictly worse than absent.** | **1.03 GPU-min** | 660.00 GPU-min |
| **during the final save** | *With slots:* the two periodic slots are deleted only **after** `out_dir` is verified complete, so the newest slot is still on disk and `latest_checkpoint` returns it. The tail since the last slot, plus the write, is all that is at risk. *Without:* nothing to fall back to inside this chunk; resume from the previous chunk, losing the entire chunk at the moment the most has been invested in it. | **1.03 GPU-min** | 660.00 GPU-min |

`restart` cost is **8.06 min** on every row and in both columns — it is the
measured-plus-assumed 484 s startup, paid whether or not anything was lost, and
it is a separate column from `lost` for that reason.

> **WHAT THE PERIODIC INTERFACE IS WORTH: a mid-chunk kill costs 1.02 GPU-min
> with it and 660.00 without — 647×, 659 GPU-minutes per kill, which is 36.6 %
> of the entire weekly quota.** Three mid-chunk kills in a week without it and
> the week is gone. **This is a measurement of another node's work, taken from
> outside it.**

### 10.2 The interface dependency, stated as the item requires

The item asks that a recovery path depending on an interface `ceq/hf/train.py`
does not yet offer be said so and priced both ways. It did not offer it at the
start of this node's run and does now, in the **working tree only**:

```
$ python -c "import inspect; from ceq.hf import train as T; \
    print('save_every' in inspect.signature(T.train).parameters)"
True
$ python -c "from ceq.hf import train as T; print(hasattr(T, 'latest_checkpoint'))"
True
```

At `git HEAD` (`ab5b485`) both are `False`. **The left column of §10.1 is not
valid against `HEAD`**, and the launch decision must be made against whichever
version is actually committed at launch. `scripts/k_cost.py` detects the
interface by `inspect.signature` at run time and prints which column it is
pricing, so this cannot silently become stale.

---

## 11. VERDICTS

| item | call |
|---|---|
| **G0.6 K-CERT** | **GREEN** — packaged, run once locally, 13/13 tests, every required half present with a printed R² or an explicit not-executable, no HALT. |
| **G0.7 K-COST** | **GREEN** — chunk table from this run's law with extrapolations marked, margin derived, quota charged for startup, ledger priced against both checkpoint interfaces. |

**Three findings a reader must carry out of this file, none of which is a
verdict on either item:**

1. **`arm_smprime` at `n = 32,768` fits neither card.** 13.969 GiB reserved on
   the certified 7.996 GiB box; 15.771 GiB projected against a 13.50 GiB T4
   budget. R2's registered `n` must move to **16,384** — the last power of two
   that fits, at 59 % of the T4 budget — or R2 must drop the workhorse arm.
   This is a **RED** on that cell, not on this item.
2. **Strict determinism is unavailable for training.** Bitwise for forward and
   replay in both regimes; `cumprod`'s backward routes through `cumsum` and
   raises under the flag. **BLOCKED** for a strict-mode training run; `RULING 1`
   already prescribes `warn_only=True` there, and this is the measurement behind
   it.
3. **`DTYPE_MODES["bf16_autocast"][0]` is 2.383 measured, 2.2 in the module** —
   8.3 % optimistic, bounded at +0.85 % on the Q3 shape and `max_batch`
   unchanged. Reported, not edited.

**Reproduction.**

```
python scripts/k_cert.py                      # -> results/k_cert_local.json
python -m pytest tests/gate0/test_g06_kcert.py -q
python scripts/k_cost.py
```

`git rev-parse HEAD` at start and end of this node: **`ab5b48547884e04258276e6e808d5a71ea65f917`**.
`git status --porcelain` empty at start; at end, five `M` and eighteen `??` from
five concurrent agents — §2.1 reads the two that could have moved a number.
**No writing git command was run.**
