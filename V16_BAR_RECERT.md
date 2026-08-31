# V16 SATURN-6 — RE-CERTIFYING THE BAR AND THE 0-STEP GATE ON CUDA

The instrument half of **D-DEVICE**. `V15_MERCURY_DEVICE.md` condition 2 came
back BLOCKED because `negation_scope.calibrate_bar` had no `device` parameter
and `--device cuda` therefore aborted in `scale/r10_capacity_sweep.py::main()`
before any journal opened. That refusal was correct — a threshold carried
across a device boundary is `MISTAKES.md` **V-22**. This node earns the
certification rather than deleting the guard.

Box: NVIDIA RTX 4060 Laptop, `torch 2.5.1+cu121`, `torch.cuda.is_available()`
True. Every number below is measured in this run on this box.

---

## LEAD CAVEAT (R11 bias: six of nine contract statements were optimistic)

**Three things are certified and one is not, and the one that is not is the
one a reader will assume from the other three.**

1. The BAR (`calibrate_bar` + `bar_verdict`) is certified on cuda.
2. The 0-STEP RED GATE is certified on cuda at R1′'s shapes.
3. `refuse_cross_device_pool` still fires, and now covers the bar record.
4. **The TRAINED `eval_nrmse` readings are NOT certified as device-comparable
   and this node did not measure them** — "nothing trains as a research
   reading" forbids it. The argument that they do not need to be is given in
   §6, and it rests entirely on condition 1's no-pooling rule actually being
   enforced. **§7 shows one place where it is not**: `scripts/v15_r1.py`, the
   runner that would execute R1′, writes `device="cpu"` as a **string
   literal** on four of its records while carrying a `--device` flag. Lift its
   abort without fixing those four lines and the guard is defeated by a
   constant, silently, on the deciding measurement of the round.

Nothing here lowers a bar. Every tolerance used was read out of `bar_verdict`'s
signature or `GATE_TOL`'s docstring — text that predates this node — and is
tabulated in §3 **before** the measured `δ` it judges.

---

## 1. WHAT CHANGED

Executable diff, comments and docstrings excluded: a signature, a call
expression, a two-line model move, and four one-statement edits in the
sweep. Everything else in the 181-line diff is the reasoning, at the point
of use. `git diff --stat`: `negation_scope.py +37`, `r10_capacity_sweep.py
+119/-37`, `test_no_cross_device_pooling.py +62`.

### `scale/negation_scope.py` — `calibrate_bar` gains a device

```python
-                  seed: int = 0, standardise: bool = True) -> dict:
+                  seed: int = 0, standardise: bool = True, device=None) -> dict:

-    x, y, f, p = (batch_fn or make_batch)(n, s, d, seed=seed)
+    x, y, f, p = (batch_fn or make_batch)(
+        n, s, d, seed=seed, **({} if device is None else {"device": device}))

     for layer in net:
         if isinstance(layer, torch.nn.Linear):
             torch.nn.init.normal_(layer.weight, 0.0, 0.5, generator=g)
             torch.nn.init.zeros_(layer.bias)
+    if device is not None:
+        net = net.to(device)
```

Nothing else in the function needed touching, and that is a property of the
function rather than luck: `feature_fn`, `oracle_fn`, `nrmse`, the flipper
clone and the standardisation all already resolve `device=x.device` or return
Python floats.

### `scale/r10_capacity_sweep.py`

- the `--device cuda` ABORT block deleted, replaced by the certification
  record and by the negative statement of what is *not* certified;
- `calibrate_bar(..., device=device)`;
- `device=a.device` added to the `t="bar"` journal record;
- `--device` help rewritten;
- the module docstring's claim *"NRMSE is not a function of the reduction
  order at this precision"* struck. It is false and the file itself refutes
  it: M-10 measured `eval_nrmse` moving `2.345e-3` on **thread count alone**,
  which is nothing but reduction order, and `refuse_cross_device_pool` exists
  because of it. A reader of a cuda run would have relied on that sentence.

### `tests/loop/test_no_cross_device_pooling.py`

Four cases added (8 total, all passing) — §5.

---

## 2. THE DEFAULT, AND HOW IT WAS VERIFIED

**`device=None`, not `torch.device("cpu")`.** `"cpu"` would be *equivalent*;
`None` is *unchanged*. The keyword is **omitted entirely** from the builder
call when it is None, so the default path's call expression is character-for-
character the one every bar reading in `results/` was taken through, and a
`batch_fn` hook that never accepted `device=` keeps working.

**Verified, not assumed, three ways.**

**(a) Bitwise, against `HEAD`, in one process.** `git show HEAD:scale/negation_scope.py`
was loaded as a second module alongside the edited one and both were called on
the same six tasks. All **30 doubles (6 tasks × 5 clauses) are bit-identical**
in their IEEE-754 hex:

| task | `predict_the_mean` | `payload_only` | `oracle` | `flipper_dependence` | `trained_two_feature` |
|---|---|---|---|---|---|
| `e3_t1` | `3ff0000000000000` | `3ff697de8b57e40d` | `0000000000000000` | `4000000000000000` | `3f80d918f5eee891` |
| `e3_t2` | `3ff0000000000000` | `3ff3a2b5cb1f2d80` | `0000000000000000` | `3ff66b7e778d1c8b` | `3f8ca2646053cc23` |
| `e3_t8` | `3ff0000000000000` | `3ff10676f22202fd` | `0000000000000000` | `3fe6d198bb242fe9` | `3f9702720f48c489` |
| `e3_t32` | `3ff0000000000000` | `3ff038169bb16e18` | `0000000000000000` | `3fd798db07dfb2e9` | `3f9924a53b6446cc` |
| `counter_squared` | `3ff0000000000000` | `3ff3bf4d2ca77bce` | `0000000000000000` | `3fd98c2b7cf8c2d8` | `3fc203f58e8eb9f5` |
| `negation_scope` | `3ff0000000000000` | `3ff6b0977b85c0ff` | `0000000000000000` | `4000000000000000` | `3f90025ab989ce35` |

(HEAD column and edited column are identical, so one column is printed.)

**(b) Through the shipped entry point.** `main()` run with `--device cpu` at
`--max-steps 0` (no cell trains) emits a `bar` record reading
`predict_the_mean 1.0`, `payload_only 1.2272241529888959`,
`flipper_dependence 1.4012436552018552`,
`trained_two_feature 0.013981593578261986` — the same doubles as (a).

**(c) `tests/loop`** — §8. Plus the three `tests/cameron` files that exercise
`calibrate_bar` through non-default hooks (`standardise=False`, scaled oracle
hooks, `counter_squared`, `c1_propagate`): **54 passed**, 199 s.

**One thing the default costs.** Every builder in `M3_TASKS` (19 entries,
checked by `inspect.signature`, not assumed) already accepts `device=`, so
always-passing would also have worked and been one line shorter. The omitted-
keyword form was chosen because it makes the byte-identity claim *structural*
— readable off the source — instead of something (a) alone has to prove.

---

## 3. THE TOLERANCES, DERIVED BEFORE THE MEASUREMENT

**M-2 discipline.** These are not chosen. Each is quoted from code that
predates this node, with the line that carries it.

| clause | tolerance | where it comes from |
|---|---|---|
| `predict_the_mean` | `abs(x - 1.0) > 1e-6` ⇒ FAIL | `bar_verdict` body, unchanged |
| `payload_only` | `x >= 1.0` ⇒ margin is `x - 1.0` | `bar_verdict` body |
| `oracle` | `x < 1e-6` | `bar_verdict` body |
| `flipper_dependence` | `abs(x - closed form) > 0.05` ⇒ FAIL | `bar_verdict(flipper_tol=0.05)` default; the 0.05 is derived in `chain_flipper_dependence`'s docstring from a 16-seed spread table and the distance to the nearest wrong rung (0.101577 at the tightest pair), neither of which is a device quantity |
| `trained_two_feature` | `x < 1.0` ⇒ margin is `1.0 - x` | `bar_verdict` body |
| 0-step RED gate | `>= 1.0 - GATE_TOL`, `GATE_TOL = 1e-3` | `train_with_checkpoints` docstring, 16 untrained CPU seeds, recorded as M-14 |

For the two one-sided clauses there is no numeric tolerance to divide by, so
`δ` is reported against **the reading's own margin to the bar** — the quantity
that would have to be crossed for the verdict to move. That is the tolerance
those clauses are actually used at.

`GATE_TOL`'s provenance was re-read rather than trusted. Its docstring cites
three minima over 16 untrained CPU seeds at `t*=2`; measured here on cpu at
`n_train=2048, n_eval=4096`:

| docstring | re-measured (cpu) | |
|---|---|---|
| softmax min 0-step eval `1.00055844` | `1.00055844126167037` (seed 2) | matches |
| `pivot_unsigned` `1.00055861` | `1.00055860806013541` (seed 2) | matches |
| `windowed_signed` `0.99997039` | `0.99997039327241743` (seed 2) | matches |

---

## 4. THE BAR: CPU AND CUDA SIDE BY SIDE

`calibrate_bar(n=N_EVAL=4096, s=S=64, d=D=24, steps=600, lr=0.02, seed=0)`
through the registered `e3_t*` hooks — the exact call `r10_capacity_sweep.main()`
makes. Separate processes per device, so the cuda run's global
`use_deterministic_algorithms(True)` cannot touch the cpu reading; `threads=8`
pinned **after** imports on both (see §7 finding F5). CUDA run has
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `use_deterministic_algorithms(True)`,
`cudnn.deterministic = True`, exactly as `main()` sets them.

| task | clause | CPU | CUDA | `δ` | tol / margin | `δ`/tol |
|---|---|---|---|---|---|---|
| `e3_t1` | `predict_the_mean` | 1 | 1 | 0 | 1e-6 | 0 |
| | `payload_only` | 1.4120774691675735 | 1.4120773501636601 | 1.190e-07 | 4.121e-01 | 2.888e-07 |
| | `oracle` | 0 | 0 | 0 | 1e-6 | 0 |
| | `flipper_dependence` | 2 | 2 | **0** | 5e-02 | 0 |
| | `trained_two_feature` | 0.0082265731871092578 | 0.0082265787654177015 | 5.578e-09 | 9.918e-01 | 5.625e-09 |
| | **VERDICT** | **CALIBRATED** | **CALIBRATED** | | exact fd `2.000000000000` | |
| `e3_t2` | `predict_the_mean` | 1 | 1.0000000843170462 | 8.432e-08 | 1e-6 | **8.432e-02** |
| | `payload_only` | 1.2272241529888959 | 1.2272241529888959 | **0** | 2.272e-01 | 0 |
| | `oracle` | 0 | 0 | 0 | 1e-6 | 0 |
| | `flipper_dependence` | 1.4012436552018552 | 1.4012437605053387 | 1.053e-07 | 5e-02 | 2.106e-06 |
| | `trained_two_feature` | 0.013981593578261986 | 0.013981630466969725 | 3.689e-08 | 9.860e-01 | 3.741e-08 |
| | **VERDICT** | **CALIBRATED** | **CALIBRATED** | | exact fd `1.414213562373` | |
| `e3_t8` | `predict_the_mean` | 1 | 1.0000000858002478 | 8.580e-08 | 1e-6 | **8.580e-02** |
| | `payload_only` | 1.0640782793345231 | 1.0640783706327031 | 9.130e-08 | 6.408e-02 | 1.425e-06 |
| | `oracle` | 0 | 0 | 0 | 1e-6 | 0 |
| | `flipper_dependence` | 0.71308552313848594 | 0.71308552313848594 | **0** | 5e-02 | 0 |
| | `trained_two_feature` | 0.022470289307256582 | 0.022470303300872813 | 1.399e-08 | 9.775e-01 | 1.432e-08 |
| | **VERDICT** | **CALIBRATED** | **CALIBRATED** | | exact fd `0.707106781187` | |
| `e3_t32` | `predict_the_mean` | 1 | 1 | 0 | 1e-6 | 0 |
| | `payload_only` | 1.0136934358369043 | 1.0136934358369043 | **0** | 1.369e-02 | 0 |
| | `oracle` | 0 | 0 | 0 | 1e-6 | 0 |
| | `flipper_dependence` | 0.368704564751893 | 0.36870459153009694 | 2.678e-08 | 5e-02 | 5.356e-07 |
| | `trained_two_feature` | 0.024553869773872392 | 0.024553861762042058 | 8.012e-09 | 9.754e-01 | 8.214e-09 |
| | **VERDICT** | **CALIBRATED** | **CALIBRATED** | | exact fd `0.353553390593` | |

**Worst `δ`/tolerance, by clause, over all four rungs:**

| clause | worst `δ`/tol | headroom |
|---|---|---|
| `predict_the_mean` | **8.580e-02** | 11.7× |
| `flipper_dependence` | 2.106e-06 | 475,000× |
| `payload_only` | 1.425e-06 | 702,000× |
| `trained_two_feature` | 3.741e-08 | 2.7e+07× |
| `oracle` | 0 (exact on both) | — |

### `chain_flipper_dependence` is device-independent BY CONSTRUCTION

The anchor holds, and the reason is structural rather than measured:

```python
def chain_flipper_dependence(s, *, t_star=None):
    t = (s - 1) if t_star is None else int(t_star)
    return 2.0 / math.sqrt(t)
```

No tensor, no device, no `torch`. It is a Python float from `math.sqrt`, and
takes no `device` parameter because there is nothing in it for a device to
change. Its four returned values are the `exact fd` column above and are
identical on both devices trivially. Its siblings
(`counter_squared_flipper_dependence`, `propagate_flipper_dependence`,
`consequence_flipper_dependence`, `e4prime_flipper_dependence`) are the same
shape.

What *is* device arithmetic is the **measured** `flipper_dependence` the bar
computes against it — a ratio of two `.abs().mean()` reductions over `n`
examples. That is the row in the table, and it moves by at most `1.053e-07`
against a `0.05` band: **2.1e-06 of the tolerance.** The band survives the
device by five orders of magnitude.

The 0.05 band's own derivation also transfers, and again structurally: it was
fitted to a **seed sampling spread**, and the seeds draw the same bytes on both
devices (§6.1). The tabulated spreads it was built from (`0.005730` at the
tightest, `0.149327` at the loosest) are properties of the draw, not of the
arithmetic.

### The tightest clause, swept

`predict_the_mean` is the only clause whose `δ` is a non-trivial fraction of
its tolerance, so it was swept rather than read at one seed: 8 seeds × 4 rungs
× 2 values of `n`, computed exactly as clause 1 computes it.

| task | `n` | max `δ` over 8 seeds | max abs(cpu−1) | max abs(cuda−1) |
|---|---|---|---|---|
| `e3_t1` | 4096 | 6.014e-08 | 6.002e-08 | 6.014e-08 |
| `e3_t1` | 32768 | **2.378e-07** | 1.191e-07 | 1.189e-07 |
| `e3_t2` | 4096 | 8.432e-08 | 8.395e-08 | 8.432e-08 |
| `e3_t2` | 32768 | 8.472e-08 | 8.472e-08 | 0 |
| `e3_t8` | 4096 | 8.580e-08 | 8.424e-08 | 8.580e-08 |
| `e3_t8` | 32768 | 8.460e-08 | 8.460e-08 | 8.404e-08 |
| `e3_t32` | 4096 | 8.442e-08 | 8.216e-08 | 8.442e-08 |
| `e3_t32` | 32768 | 8.451e-08 | 8.439e-08 | 8.451e-08 |

Two readings of this table, and the second is the one that matters.

- **The gap grows with `n`.** `6.0e-08` at `n=4096` → `2.4e-07` at `n=32768`,
  i.e. **23.8% of the 1e-6 tolerance, 4.2× headroom**, at the largest `n` R2
  would use. The bar is only ever *called* at `n = N_EVAL = 4096`
  (`r10_capacity_sweep.py`) or `n = a.n_eval` (`scripts/v15_r1.py`, default
  4096), where the worst is 8.6% and headroom is 11.7×. Calling the bar at
  `n = 32768` is not currently reachable, and this row is why it should not be
  made reachable without re-reading it.
- **It is fp32 accumulation, not a device effect.** `abs(cpu − 1.0)` reaches
  `1.191e-07` and `abs(cuda − 1.0)` reaches `1.189e-07` — the same size. This
  clause is an algebraic identity (`nrmse(mean, y) ≡ 1`) evaluated in float32,
  and **the CPU consumes 12% of the 1e-6 budget at `n=32768` on its own**. The
  device is not the thing eating this tolerance; `n` is.

---

## 5. `refuse_cross_device_pool` — still fires, and now covers the bar

The four V15 cases still pass unchanged (must-fire on 8 cpu + 8 cuda cells;
single-device control; missing-field-reads-cpu; empty pool). Four added, for
what this node touched:

| test | what it pins |
|---|---|
| `test_must_fire_a_pool_of_bar_records_spanning_two_devices_raises` | The bar is now a per-device measurement. The two records carry the **measured** cpu/cuda `trained_two_feature` from §4 (`0.013981593578261986` vs `0.013981630466969725`) — a pair differing only in the last figures, i.e. exactly what a value-comparing guard waves through. |
| `test_a_bar_record_pools_with_the_cells_it_certified` | Control. A cuda bar + its own cuda cells must **not** refuse, or the certified path is unrunnable rather than safe. |
| `test_a_cuda_cell_may_not_be_read_against_a_cpu_bar` | The exact shape the V15 abort was protecting against, now expressed as data instead of a `return 1`. |
| `test_the_analytic_ceiling_row_is_the_documented_false_positive` | See below. |

**One false positive found and pinned rather than hidden.** `main()` writes a
`t="ceiling"` record with **no** `device` field, deliberately —
`sqrt((t*-1)/t*)` is a closed form and stamping a device on it would assert it
had been measured on one. Under the missing-field-reads-`"cpu"` rule it will
refuse against the cuda cells of *its own journal*. A caller must filter by
`t` first. The false positive is loud, which is the right direction, but a
consumer handing the guard a whole journal file will hit it. Documented at the
function and pinned by the fourth test, so a later change that starts
device-stamping the ceiling has to come to that test and say why.

`tests/loop/test_no_cross_device_pooling.py`: **8 passed**, 2.40 s.

---

## 6. THE 0-STEP RED GATE AT R1′'s SHAPES

`CEQ_V16_CONTRACT.md` PART IV: *"R1′ THE RE-RUN. BED-M, `t*=2`, `n=2048`,
phase gates, same seeds"*, `N = 8`. `scripts/v15_r1.py` defaults fix the rest:
`n_eval = 4096`, `seeds 0..7`, `threads 8`, `S=64, D=24`. So the gate was read
at **`t*=2, n_train=2048, n_eval=4096, s=64, d=24`, all four arms this file can
score × 8 seeds = 32 cells, 64 readings, per device** — via
`train_with_checkpoints(..., rungs=[])`, which is the shipped function with
zero optimizer steps.

**All 32 cells PASS the gate on both devices.** Summary:

| | value |
|---|---|
| max `abs(cpu − cuda)`, `nrmse0_train` | `1.617694e-07` |
| max `abs(cpu − cuda)`, `nrmse0_eval` | `1.678926e-07` |
| that as a fraction of `GATE_TOL = 1e-3` | **`1.679e-04`** |
| smallest margin any of the 64 readings holds above the gate line `1.0 - GATE_TOL` | `9.703933e-04` |
| margin ÷ device gap | **5,780×** |

The binding cell is `windowed_signed` seed 2 — the same cell `GATE_TOL`'s
docstring names as the one that would abort a run against a bare `1.0`. It
reads `0.99997039327241743` on cpu and `0.99997056006842744` on cuda: **CUDA
sits 1.67e-07 further ABOVE the line than CPU does.** The CPU derivation is
therefore the conservative one of the two, and moving to cuda does not loosen
`GATE_TOL` at this cell.

Per-arm worst case over the 8 seeds:

| arm | max `δ` train | max `δ` eval | min margin above line | all pass |
|---|---|---|---|---|
| `softmax` | 8.460e-08 | 8.449e-08 | 1.5584e-03 (seed 2 eval) | yes / yes |
| `pivot_unsigned` | 8.426e-08 | 1.679e-07 | 1.5586e-03 (seed 2 eval) | yes / yes |
| `windowed_signed` | 8.422e-08 | 1.671e-07 | **9.7039e-04** (seed 2 eval) | yes / yes |
| `pivot_signed` | 1.618e-07 | 1.668e-07 | 1.5407e-03 (seed 2 eval) | yes / yes |

Seed 2 is the binding seed on every arm, which is the seed `GATE_TOL`'s
docstring already names as the tail it was sized to admit.

### V15's bit-identity claim: reproduced, and shown not to generalize

`V15_MERCURY_DEVICE.md` (d) records the 0-step gate **bit-identical** on both
devices at `t*=2, n=512, n_eval=512, seed=0, softmax`. Re-run here:

| shape | cpu train / eval | cuda train / eval | |
|---|---|---|---|
| `n=512, n_eval=512` | `1.00290663131789071` / `0.99932432996255216` | identical | **BIT-IDENTICAL** — V15's table reproduced digit for digit |
| `n=2048, n_eval=4096` | `1.00363199447569662` / `1.00335073096925420` | `1.00363191026599052` / `1.00335081464794218` | differs by `8.421e-08` / `8.368e-08` |

**Bit-identity is a property of that tiny shape, not of the gate.** At R1′'s
own shapes it is gone. The certification above is therefore *"inside a
pre-derived tolerance by 5,780×"*, which is a different and weaker claim than
*"bitwise"* — and it is the true one.

### 6.1 The corpus is the same draw on both devices

The load-bearing fact under everything above. Every `M3_TASKS` builder draws
from `torch.Generator(device="cpu")` and only then `.to(x.device)`, so the
tensor **bytes** should not depend on the device. Verified by sha256 of the
raw buffer, 12 shapes × (x, y):

`e3_t{1,2,8,32}` × `n ∈ {2048, 4096, 32768}` at `s=64, d=24, seed=0` —
**every x and every y byte-identical on cpu and cuda**, and `f`, `p` equal.
(e.g. `e3_t2, n=32768`: x `9855e52a8301c08b`, y `9b70161448e8dc3e` on both.)

Consequence: every CPU-measured **sampling-spread** table in this tree
transfers to cuda by construction rather than by luck, because it is a
property of the draw. Every CPU-measured **arithmetic** constant does not, and
§7 lists them.

### 6.2 CUDA is reproducible run-to-run, so `by_seed`'s 1e-9 rule holds

`it11_verdict.by_seed` raises on a same-seed, same-`threads` disagreement above
`1e-9`. If cuda were not bitwise reproducible, a legitimate repeat would trip
it. Measured with the shipped determinism flags on:

- 0-step gate, `t*=2, n=2048/4096, seed 0, softmax`, **3 repeats in one
  process**: bitwise identical.
- **150 trained steps**, same cell, 2 repeats: `eval_nrmse` `0.97340022953138716`
  both times, `abs(A−B) = 0.000e+00`. (Instrumentation for determinism only —
  it is one seed at a step count no R1′ rung reads, and it is not a reading.)

### 6.3 `bootstrap_ci` runs on device tensors

It is on the credited path and takes CPU-generated indices into device
tensors, which is not obviously legal. It works: `pred[idx]` with a CPU
`idx` and a cuda `pred` is accepted, and under `use_deterministic_algorithms(True)`
it does not raise. On a 4096-row eval set: cpu `[1.0000002515257553,
1.0005638268514907]`, cuda `[1.0000002468644105, 1.0005638268514907]` — `hi`
bitwise identical, `lo` apart by `4.66e-09`. It costs 400 host→device index
transfers per call, which is a performance note, not a correctness one.

### 6.4 The shipped entry point, both devices, end to end

`main()` run at `--max-steps 0` (every cell skipped, nothing trains) with
`ROOT` redirected to a scratch dir so no file in `results/` was touched:

```
cpu   exit=0   header device=cpu  deterministic_algorithms=False
               bar    device=cpu  BAR CALIBRATED  predict_the_mean 1.0
cuda  exit=0   header device=cuda deterministic_algorithms=True
               bar    device=cuda BAR CALIBRATED  predict_the_mean 1.0000000843170462
```

`--device cuda` no longer returns 1. The cpu bar record is bit-identical to
§2(a)'s HEAD column.

**Determinism cost is V15's measurement, not re-taken here.**
`V15_MERCURY_DEVICE.md` (b.3) measured `use_deterministic_algorithms(True)` at
**0.921×** — no slowdown — over 20 timed steps. The setting is still enabled
unconditionally on the cuda path and still declared live in the header via
`torch.are_deterministic_algorithms_enabled()`, confirmed `True` above.

---

## 7. EVERYTHING ELSE THAT IS SILENTLY CPU-ONLY

Each is a V-22 in waiting: a constant established on one system and read on
another. **None is in this node's write scope.** Ordered by what they would
cost if left.

### F1 — `scripts/v15_r1.py` hardcodes `device="cpu"` on four records — **HIGHEST**

That file is R1′'s own runner (`--arms arm_pl softmax`, `--n-train 2048`,
`--n-eval 4096`, `--seeds 0..7`, `--threads 8` — the exact shapes §6
certified). It **has a `--device` flag** and it **calls
`refuse_cross_device_pool` on the collected rows before any verdict**. And it
writes the string literal `device="cpu"` at lines **257, 367, 374 and 424** —
header, manifest, cell and summary records.

So if its abort (lines 233-238) were lifted, a `--device cuda` run would journal
`device="cpu"` on every record, `refuse_cross_device_pool` would see one
device, and a cuda pool would merge with cpu cells **silently**, on the
deciding measurement of the round. The guard would be defeated by a constant.
Its abort is currently the only thing preventing that. **Do not lift that abort
without first changing those four literals to `a.device`.**

Its abort text is also now stale — it says *"`calibrate_bar` is CPU-only"*,
which this node has made false.

### F2 — `it11_verdict.THREAD_FLOOR = 2.345e-3` gates `delta_eq()` — **HIGH**

```python
THREAD_FLOOR = 2.345e-3   # measured, three cross-thread pairs; see `by_seed`
...
if d < 2 * THREAD_FLOOR:
    raise ValueError(f"Delta_eq = {d:.6g} is below 2x the reduction-order floor ...")
```

This is a **CPU thread-count** reduction-order floor, measured on cpu across
`threads=6` and `threads=8` journals. On cuda, thread count does not name the
reduction lane at all — the GPU's accumulation order is not a function of it —
so the analogous floor is a different quantity that **has never been measured**.
A cuda round that calls `delta_eq()` is admitting or refusing its equivalence
margin against another system's constant. That is V-22 exactly, one level up
from a decision threshold, in the same place V-22's own entry puts it.

Cost to fix: the cuda reduction-order floor is measurable at 0 training steps
if it is defined as run-to-run spread (§6.2 says it is **0** with the shipped
determinism flags, which would make the floor vacuous and the guard
inapplicable rather than wrong) — but if it is defined as *cpu-vs-cuda* spread
at the deciding step count, measuring it requires trained cells, which this
node may not run.

### F3 — `it11_verdict.by_seed` buckets by `threads` only — **HIGH**

Already flagged as a limit by `V15_MERCURY_DEVICE.md`. Worth restating with
the new fact: on cuda, `threads` is a **decoy field**. It is journalled, it is
honoured by torch, and it governs nothing about the reduction. Every cuda row
lands in one `threads` bucket regardless of the lane that produced it, and a
cuda row and a cpu row at the same `threads` land in the *same* bucket.
`refuse_cross_device_pool` is still not wired into `by_seed`.

### F4 — `ceq/arm_pl.py:238` and `ceq/arm_phase.py:384` hardcode `"device": "cpu"` into the **identity manifest** — **HIGH**

`scale/identity_manifest.CONFIG_FIELDS` carries `device` as a first-class
field, and `V15_MERCURY_DEVICE.md` (d) demonstrates the mechanism working: the
cpu and cuda manifests hash differently. Both arm modules defeat it with a
literal. A cuda run of either arm produces a manifest **hash identical to the
cpu one**, i.e. a manifest asserting the run happened somewhere it did not.
`arm_phase` is the arm `CEQ_V16_CONTRACT.md` names for R1′.

`ceq/arm_phase.py` additionally calls `.numpy()` on its route-check tensors
(`:257`, `:262`) — those raise on a cuda tensor — and `ceq/arm_pl.py:296` does
`a.detach().cpu().numpy()`. Whether either arm's forward runs on cuda at all
is **not established by this node** (`ceq/**` is out of scope and R1′ may not
be run).

### F5 — `scale/m3_capability.py:64` re-pins threads at IMPORT time — **MEDIUM**

`torch.set_num_threads(2)` at module scope. Any caller that pins threads
*before* importing it silently gets 2. It bit this node's own first probe:
`torch.set_num_threads(8)` then `import scale.m3_capability` reads back 2.
`r10_capacity_sweep.main()` and `scripts/v15_r1.py` happen to pin after their
imports, so the shipped paths are correct — but M-10 measured `eval_nrmse`
moving `2.345e-3` on thread count, so a probe or script that pins first has
its numbers moved by an import. 37 other modules in `scale/` do the same.
Line 15 of that file also still declares `CPU ONLY. ... calibrate_bar` — now
false.

### F6 — `scale/vram_gate.py:36` and `scale/r10_it8_pricing.py:25` assert the sweep spends no VRAM — **MEDIUM**

Both say `scale/r10_capacity_sweep.py`'s harness *"is CPU-only by
construction (`m3_capability.py:261` — 'no .cuda() anywhere in this file')"*
and price only host RSS. **That is now false** — this node just made
`--device cuda` produce a scored verdict. A cuda sweep at `n=32768` will
spend VRAM that `vram_gate` is not being asked about, on a 7.996 GiB card, and
`vram_gate` exists precisely so *"the next OOM will look like a sizing bug and
will not be one."* NEPTUNE is pricing D-DEVICE concurrently
(`scripts/v16_device_probe.py`) and this line belongs in that pricing.

### F7 — pilot constants read as resolutions on a cuda round — **LOW/MEDIUM**

`scale/e_ladder.py:96 RESOLUTION_13 = 0.027260` (pre-registered from a CPU
pilot SD, correctly frozen with provenance per M-2 — but the pilot was CPU);
`scale/eprocess.py:223 SD_PAIRED = 0.056889`, `:226 EFFECT_FLOOR = 0.05`,
`:580 PILOT_GATE = 0.10`. All are **seed** spreads, and §6.1 shows the seeds
draw identical bytes on both devices, so these transfer by construction the
way the flipper band does — provided the spread they describe is seed spread
and not arithmetic spread. Worth a line in whichever document reads them, not
a code change.

### F8 — `negation_scope.py:270, 280` — `E2_T_STAR`'s measured crossing (`k=16`) and `E2_STEPS = 600` — **LOW**

Both say *"this machine"* and both were measured on cpu. `E2_STEPS = 600` is
the step budget `calibrate_bar`'s E2 control needs to clear the bar, chosen as
*"the smallest of those three that passes"* — i.e. a value sitting on a
threshold. `e2_consequence` is not on R1′'s or R2's path and has never been
trained (FINDINGS A7), so nothing this round reads it. If E2 is ever run on
cuda, 600 is a cpu measurement of a margin and needs re-reading.

---

## 8. `tests/loop` BEFORE AND AFTER

Measured here, both times, `python -m pytest tests/loop -q --no-header -p no:cacheprovider`.
**The count in circulation has been stale twice (515 quoted, 517 measured,
518 with a new module); it is 518 at this commit.**

| | failed | passed | time |
|---|---|---|---|
| before (measured at session start, working tree clean of this node's edits) | **15** | **518** | 34.16 s |
| after | **15** | **522** | 42.38 s |

The failure **set** is the invariant, and it is **identical line for line** —
diffed as sorted `FAILED`/`ERROR` lines, not eyeballed:
`test_conftest_import_is_order_dependent.py` ×10,
`test_corpus_is_recoverable_and_verifiable.py` ×2,
`test_every_boundary_node_can_propagate.py` ×1,
`test_manifest_refuses_an_absence_it_has_not_earned.py` ×1,
`test_the_bar_control_is_scored_out_of_sample.py` ×1. **No new failure.**

Passed `518 → 522`, `+4` = the four cases added to
`tests/loop/test_no_cross_device_pooling.py`. Nothing else moved.

Additionally, the three `tests/cameron` files that drive `calibrate_bar`
through non-default hooks: **54 passed**, 199.25 s.

---

## 9. THE DECISION

**CUDA IS CERTIFIED FOR THE INSTRUMENT, NOT FOR THE READINGS.**

### Certified

| artifact | shapes | tolerance (pre-derived) | measured `δ` | verdict |
|---|---|---|---|---|
| `negation_scope.calibrate_bar` + `bar_verdict`, all five clauses | `e3_t{1,2,8,32}`, `n=4096, s=64, d=24, steps=600, lr=0.02, seed=0` — the exact call `r10_capacity_sweep.main()` makes | `bar_verdict`'s own: 1e-6, 1e-6, ≥1.0, 0.05, <1.0 | worst `8.580e-08` = **8.6%** of the binding clause's tolerance | **CALIBRATED on both devices, all four rungs** |
| `predict_the_mean` (binding clause), swept | 8 seeds × 4 rungs × `n ∈ {4096, 32768}` | 1e-6 | `8.58e-08` at `n=4096`; `2.38e-07` (23.8%) at `n=32768` | inside at both; `n=32768` is not a shape the bar is called at |
| 0-step RED gate | `t*=2, n_train=2048, n_eval=4096, s=64, d=24`, arms `softmax`/`pivot_unsigned`/`windowed_signed`/`pivot_signed` × seeds 0–7 = 32 cells, 64 readings | `GATE_TOL = 1e-3` | `1.679e-07` = **1.7e-04** of `GATE_TOL`; smallest margin above the line `9.704e-04` = **5,780×** the gap | **PASSES on both devices, all 32 cells** |
| corpus determinism | 12 shapes × (x, y) | bytewise | 0 | **byte-identical draw on both devices** |
| cuda run-to-run reproducibility | 0-step ×3, 150-step ×2 | `by_seed`'s 1e-9 | 0 | **bitwise** |
| `refuse_cross_device_pool` | cells, bars, mixed, missing-field, empty, ceiling-row | must raise / must not raise | — | **8/8 tests pass; must-fire fires** |

`scale/r10_capacity_sweep.py --device cuda` now runs and returns 0.

### NOT certified

1. **Trained `eval_nrmse` device-comparability at the deciding step counts.**
   Not measured, and this node may not measure it — "nothing trains as a
   research reading". `V15_MERCURY_DEVICE.md` measured a `~1.3e-5` separation
   at 150 steps / `n=512` and flagged, correctly, that whether it compounds at
   9600 steps / `n=32768` is unknown. **It is still unknown.**

   **Why it does not block the certification.** Condition 1 forbids pooling a
   cuda cell with a cpu cell *at all*, and the contract says the round runs on
   the certified device **only**. A single-device round never needs the two
   devices' trained readings to agree; it needs its *instrument* — the bar and
   the gate — measured on the device its cells were taken on, which is what
   §4 and §6 deliver. Everything else a cell is scored against is a closed
   form with no device in it: the absolute bar `1.0`, the 1-hop ceiling
   `sqrt((t*-1)/t*)`, `floor₁`, `ĥ = t*(1 − NRMSE²)`.

   **What would break that argument** is any path where a cuda reading meets a
   cpu one without the guard seeing it. F1 is exactly such a path.

2. **`Δ_eq`'s admissibility floor on cuda** — F2. `delta_eq()` refuses a
   margin below `2 × THREAD_FLOOR`, and `THREAD_FLOOR` is a cpu quantity. Any
   R1′/R2 equivalence or resolution statement computed on a cuda round reads
   it. Out of write scope; named here because it is the next V-22 to land.

3. **Whether `arm_pl` / `arm_phase` execute on cuda at all** — F4. Not
   established. `ceq/**` is out of scope and R1′ may not be run. The bar and
   gate are certified for the arms `r10_capacity_sweep.py` can construct;
   `arm_phase`, which `CEQ_V16_CONTRACT.md` names for R1′, is not among them
   and calls `.numpy()` on tensors in its route check.

### The blocker, named, and what it costs

**R1′ cannot be run on cuda today, and the reason is no longer
`calibrate_bar`.** It is `scripts/v15_r1.py`, which is out of this node's write
scope and which needs, in order:

1. its four `device="cpu"` literals (lines 257, 367, 374, 424) changed to
   `a.device` — **this is not optional and must come first**, because
   `refuse_cross_device_pool` is blind without it (F1);
2. `calibrate_bar(..., device=device)` threaded (one argument — the parameter
   now exists);
3. `device` threaded to its `batch_fn` calls and its arm construction, using
   `train_with_checkpoints`'s construct-then-`.to()` ordering;
4. the stale abort at lines 233-238 deleted, and its docstring's *"CPU ONLY, BY
   DESIGN … `calibrate_bar` is CPU-only"* rewritten;
5. `ceq/arm_pl.py:238` / `ceq/arm_phase.py:384`'s manifest `"device"` literals
   made live (F4), or the manifest hash lies about where R1′ ran;
6. a check that `arm_phase`'s complex64 forward and its `.numpy()` route probe
   survive a cuda tensor under `use_deterministic_algorithms(True)`.

Steps 1–4 are a handful of lines in one file. Steps 5–6 touch `ceq/**` and are
someone's node. **None of them is a re-certification** — the two artifacts
D-DEVICE named are certified above. They are plumbing, plus one guard that is
currently being defeated by a string constant.

---

## LIMITS

- The bar's trained clause (`trained_two_feature`) is certified at **seed 0
  only**, because `seed=0` is the only seed the shipped call ever uses
  (`calibrate_bar`'s default; neither caller overrides it). The cheap clauses
  were swept over 8 seeds; clause 5 was not, and 4 rung-cells × 600 steps × 2
  devices is what a seed sweep of it would cost.
- `predict_the_mean`'s headroom is **4.2×** at `n=32768` against 11.7× at the
  shipped `n=4096`, and the gap grows with `n`. The bar is not reachable at
  `n=32768` today. If it is ever called there, re-read that row first.
- The 0-step gate was certified on the four arms `r10_capacity_sweep.py` can
  construct. **`arm_phase`, the arm the contract names for R1′, is not one of
  them** and was not tested.
- `use_deterministic_algorithms(True)`'s `0.921×` cost is **V15's
  measurement, cited, not re-taken here.** What was verified is that the flag
  is live on the cuda path (`deterministic_algorithms: true` in the header).
- Run-to-run cuda determinism (§6.2) was measured **in one process**. Two
  separate processes were not compared; a cuBLAS workspace or autotune
  difference across process boundaries would not have been caught.
- `MISTAKES.md` showed `M` in `git status` partway through this node's run and
  was clean again by the end — a concurrent node is writing it, and F1–F8 were
  gathered against a tree that was moving. Nothing here edits it; F1–F8 are
  reported for whoever owns that file to adjudicate, and each names its file
  and line so it can be re-checked against whatever landed.
- `scripts/v16_device_probe.py` (NEPTUNE, D-DEVICE pricing) ran concurrently
  and states in its own header that it observed `calibrate_bar` gaining a
  `device=` parameter mid-run and that it deliberately does not print the five
  clause values. It measures wall clock and bytes; this file measures the
  clauses. There is no overlap and no contradiction between the two.
- No `results/` journal was written. The `main()` smoke test in §6.4
  redirected `ROOT` to a scratch directory. No R1′, R2 or deciding cell was
  run, and no number above is a capability reading.
