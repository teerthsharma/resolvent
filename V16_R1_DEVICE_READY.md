# V16 MERCURY-5 — R1′'s RUNNER, ITS MANIFESTS AND ITS ARM, MADE DEVICE-AWARE

`V16_BAR_RECERT.md` certified the instrument on CUDA and then named what still
blocks R1′. F1 and F4 were the two that touch the deciding measurement's own
records. Both are closed here, and the arm `CEQ_V16_CONTRACT.md` names for R1′
(`ceq/arm_phase.py`) now runs on device with its five binds re-measured there.

Box: NVIDIA RTX 4060 Laptop, `torch 2.5.1+cu121`, `torch.cuda.is_available()`
True. Every number below is measured in this run on this box.

**Nothing was trained.** Every runner invocation below is `--steps 0`, which is
the existing 0-step RED gate and nothing more. No `results/` journal was
touched: `ROOT` was redirected to a scratch directory for both smoke runs.

---

## LEAD CAVEAT — R1′ IS NOT UNBLOCKED, AND THE REMAINING BLOCKER IS NEW

The four literals are gone and the guard fires. But a CUDA R1′ now has to pick
one of two things it cannot have both of, and this node found the conflict
rather than resolving it:

**`cumsum_cuda_kernel` has no deterministic implementation in torch 2.5.1, and
`cumsum` is the whole of both arms.** `use_deterministic_algorithms(True)` — the
flag `scale/r10_capacity_sweep.py::main()` sets unconditionally on its cuda path,
and the regime `V16_BAR_RECERT.md` §4 certified the bar under — makes
`arm_pl.scan`, `arm_phase.scan_phase`, `key_bias`, `phase_factor`, `normalizer`,
`winding`, `winding_matrix` and both arms' `operator` **raise** on cuda. So the
bar is certified in a regime the arm cannot execute in. §6 measures both sides.
`scripts/v15_r1.py` does not set the flag (it never did); it now **journals**
it, so a reading says which regime produced it. Deciding the trade is not this
node's.

---

## 0. BASELINE, MEASURED HERE, BEFORE ANY EDIT

`python -m pytest tests/loop -q -p no:randomly`, working tree clean of this
node's edits:

| | failed | passed | time |
|---|---|---|---|
| **before** | **15** | **522** | 33.52 s |
| **after** | **15** | **522** | 92.35 s |

`522` is the count `V16_BAR_RECERT.md` §8 left behind, so the circulating figure
is finally not stale. The failure **set** is the invariant and it is **identical
line for line** — diffed as sorted `FAILED`/`ERROR` lines, `diff` returns empty:

```
test_conftest_import_is_order_dependent.py          × 10
test_corpus_is_recoverable_and_verifiable.py        ×  2
test_every_boundary_node_can_propagate.py           ×  1
test_manifest_refuses_an_absence_it_has_not_earned.py ×  1
test_the_bar_control_is_scored_out_of_sample.py     ×  1
                                                    ---- 15
```

**No new failure.** Passed count unmoved at 522 because nothing was added to
`tests/loop`; the additions are in `tests/arm_phase`, §5.

(The wall-clock difference is scheduling, not work: the "before" run had the box
to itself and the "after" run shared it. No collected test changed.)

---

## 1. F1 — THE FOUR LITERALS, WITH VERIFIED LINE NUMBERS

`V16_BAR_RECERT.md` reported lines 257 / 367 / 374 / 424. Verified against the
file as it stood at the start of this node: **all four correct**. After the edit
the file has moved; the table gives both.

| site | was (verified) | is now | record | what it now writes |
|---|---|---|---|---|
| header | `257` `device="cpu"` | **`309`** `device=dev` | `t="header"` | the run's `--device`, plus a new `deterministic_algorithms` field |
| cell | `367` `device="cpu"` | **`430`** `device=dev` | `t="cell"` | **the row `refuse_cross_device_pool` actually reads** |
| manifest base | `374` `device="cpu"` | **`441`** `device=dev` | `identity_manifest.manifest(...)` input | the `CONFIG_FIELDS` `device` entry, live |
| seed aggregate | `424` `device="cpu"` | **`492`** `device=dev` | `t="agg"` | the device the summarised cells were taken on |

`dev = a.device`, bound at **line 284**, immediately where the V15 abort used to
sit (lines 233-238, deleted).

### What the guard was actually being fed

`scripts/v15_r1.py` already had `--device`, and already called
`refuse_cross_device_pool(rows)` at what is now line 469. Every row it saw said
`cpu`. Reproduced by re-stamping the literal onto the real journals of §2:

```
devices in pool = ['cpu']  ->  RETURNED.
The identical 8 rows, 4 of them measured on cuda, pooled SILENTLY.
```

That is `V-22`'s shape — a constant standing in for a measured quantity — sitting
on the round's deciding measurement, and it read as a live guard.

### Everything else threaded, and the ordering that matters

| line | change |
|---|---|
| 284 | `dev = a.device`; the abort at 233-238 deleted |
| 316 | `bind_check(emit, device=dev)` — the bind is re-measured on the run's own device |
| 347 | `calibrate_bar(..., device=dev)` — the parameter `V16_BAR_RECERT.md` §2 added |
| 352 | `t="bar"` record carries `device=dev` (the guard covers bars, §5 of that file) |
| 362, 371 | `batch_fn(..., device=dev)` for the eval and per-seed train draws |
| 373 | `train_one(..., device=dev)` |
| 127-128 | `train_one` does `make_arm(...)` **then** `.to(device)` |
| 412 | the 0-step probe control's rebuilt model `.to(dev)` |
| 214-215 | `probe()`'s intercept column is `torch.ones_like(feat[:, :1])`, not `torch.ones(...)` — the one site that would have raised on a cuda feature tensor |
| 247 | `bind_check`'s `zero_q` takes `device=b.device` |
| 517, 537 | `t="contrast"` and `t="probe"` carry `device=dev` |
| 570-572 | `t="wall"` carries `cuda_max_memory_allocated_gib` on cuda beside the host peak working set |

**CONSTRUCT, THEN MOVE, EVERYWHERE.** `torch.manual_seed(seed)` → build on host →
`.to(device)`. A device generator draws different bytes; this ordering is the one
`V16_BAR_RECERT.md` §6.1 verified byte-identical, and it is why the two smoke runs
in §2 agree to the last digit on `eval_nrmse`.

### The docstring

`"CPU ONLY, BY DESIGN. --device cuda aborts: calibrate_bar is CPU-only"` is
struck. It is false as of `V16_BAR_RECERT.md` §4. Replaced by a statement of what
the literals were doing, the construct-then-move rule, and the determinism
caveat of the LEAD CAVEAT.

### Every journalled record now names its device

`--device cuda --steps 0 --seeds 0 1 --n-train 512 --n-eval 512`, `ROOT`
redirected, exit `0`:

```
t=header  device='cuda'      t=cell  device='cuda'   t=agg       device='cuda'
t=bind    device='cuda' ×2   t=cell  device='cuda'   t=agg       device='cuda'
t=bar     device='cuda'      t=cell  device='cuda'   t=contrast  device='cuda'
                             t=cell  device='cuda'   t=probe     device='cuda'
                                                     t=wall      device='cuda'
```

13 of 13. Before this node, 8 of those 13 said `cpu` unconditionally and 5 said
nothing.

---

## 2. THE GUARD, FIRING ON A GENUINELY MIXED POOL

Not synthetic rows. Two real runs of the shipped entry point, one per device,
same shapes and seeds, journals read back off disk. **The two devices' readings
for the same cell are bit-identical at this shape**, which is the whole point:
a value-comparing guard has nothing to see.

| cell | cpu `eval_nrmse` | cuda `eval_nrmse` |
|---|---|---|
| `arm_pl:t2:n512:seed0` | `1.0144792053138267` | `1.0144792053138267` |
| `arm_pl:t2:n512:seed1` | `1.026939584619789` | `1.026939584619789` |
| `softmax:t2:n512:seed0` | `0.9993243299625522` | `0.9993243299625522` |
| `softmax:t2:n512:seed1` | `1.0254930842712424` | `1.0254930842712424` |

| pool | devices seen | required | result |
|---|---|---|---|
| 4 cpu cells | `['cpu']` | must not refuse | returned — **PASS** |
| 4 cuda cells | `['cuda']` | must not refuse | returned — **PASS** |
| **4 cpu + 4 cuda cells** | `['cpu', 'cuda']` | **must refuse** | **`ValueError: refuse to pool cells across devices ['cpu', 'cuda']`** — **PASS** |
| 3 cpu + 1 cuda cell | `['cpu', 'cuda']` | must refuse | **raised** — **PASS** |
| cpu cells + one cuda `t="bar"` | `['cpu', 'cuda']` | must refuse | **raised** — **PASS** |
| the same 8 rows, literal re-stamped | `['cpu']` | — | **returned silently**: the defect, reproduced |

The must-fire case fires on rows that no value comparison could separate. That
is the demonstration `V-9` asks for: the guard is not repaired until it is shown
firing.

`refuse_cross_device_pool` itself is **unchanged**. `scale/**` was not touched.

---

## 3. F4 — TWO MANIFESTS, SAME CELL, TWO DEVICES, DIFFERENT HASHES

`scale/identity_manifest.CONFIG_FIELDS` has carried `device` all along. Both arm
modules were filling it with a literal, so a cuda cell hashed identically to a
cpu one — a manifest asserting the run happened somewhere it did not.

| file | was | is now |
|---|---|---|
| `ceq/arm_phase.py` | `:384` `"device": "cpu"` | **`:415`** `"device": a.device.type` |
| `ceq/arm_pl.py` | `:238` `"device": "cpu"` | **`:255`** `"device": b.device.type` |

`.type` and not `str(device)`: `"cuda"`, not `"cuda:0"`, so the value is the
vocabulary `--device` uses and the bucket `refuse_cross_device_pool` reads.

`label_cell(*draw(seed=15, s=8), seed=15)`:

**`ceq/arm_phase.py`** — the arm the contract names for R1′

| | cpu | cuda |
|---|---|---|
| `record["device"]` | `'cpu'` | `'cuda'` |
| `manifest.values["device"]` | `'cpu'` | `'cuda'` |
| `manifest.config` | `eee37afcf1505bb6e0e5263a02e621e8cf2633a95b2be1ca09a949a31cfcebae` | `a4803e64dedca856cff548072b405b67387667857461d90dc28a6a4538b7ac5d` |
| **`manifest.hash`** | **`128d99eef9682fd372cc06bf2cb60a8bc3ab9340e9ef6e9a81b702356ca3b071`** | **`b647791e9252d2b3d15fb3f03a52cd553808d9f4b23dd29bf6009289a1eed20a`** |
| `residual` | `9.155133597044475e-16` | `9.155133597044475e-16` |

**`ceq/arm_pl.py`**

| | cpu | cuda |
|---|---|---|
| **`manifest.hash`** | **`c397733af9a20a80be5d5a667c2d50fd51ff0b24d7b15b66a134ce23caf98c2f`** | **`3a2eee36bd6368233a34723b1676013a661112b12d1775a9e7c2346efcadb165`** |
| `residual` | `6.661338147750939e-16` | `2.220446049250313e-16` |

**Note what the residual column says about why this matters.** For `arm_phase`
the two devices return the identical residual to the last bit. A reader
comparing values alone would conclude the two records are the same measurement.
The hash is the only thing that says otherwise, and until this edit it agreed
with them.

**NO PUBLISHED HASH MOVED.** On cpu, `b.device.type` **is** the string `"cpu"` —
the literal it replaces. Every stored cpu manifest hashes exactly as before, and
`tests/arm_pl` (17 tests, not in this node's write scope, unedited) passes
unchanged.

---

## 4. F-ARM — `ceq/arm_phase.py` ON DEVICE

### The `.numpy()` sites, verified and fixed

`V16_BAR_RECERT.md` reported `:257` and `:262`. Both correct, both inside
`band_modulus`, and there was a third on the same line as the first.

| was | is now | change |
|---|---|---|
| `:253` `torch.from_numpy(rng.uniform(...))` | `:261` `.to(device)` | the prefix route's phases |
| `:254` `torch.full((n,), ..., dtype=DTYPE)` | `:262` `device=device` | the prefix route's moduli |
| `:257` `m.numpy()`, `th.numpy()` | `:265` `m.cpu().numpy()`, `th.cpu().numpy()` | **raised `TypeError` on a cuda tensor** |
| `:262` `prefix.numpy()` | `:272` `prefix.cpu().numpy()` | same |
| `:370` `torch.zeros(n, 1, dtype=DTYPE)` | `:389` `device=a.device` | `label_cell`'s `q = 0` |
| `draw`, `band_draw` | `:311`, `:334` | `device=None` parameter, `.to(device)` **after** `default_rng` |

**The product route stays on the host on purpose.** numpy has no cuda, and the
whole point of the second route is that it shares no code with the first (`V-3`).
It is fed `.cpu()` copies, so on a cuda run it checks *cuda prefix scan* against
*host cumulative product* — a stronger reading than the cpu run's, not a weaker
one. `max_route_gap` is `1.5654144647214707e-14` on **both** devices.

### The binds, CPU and CUDA side by side

All five are `V15_ARM_PHASE.md`'s published claims, re-measured, nothing relaxed.

| bind | claim | CPU | CUDA | verdict |
|---|---|---|---|---|
| **1** `\|a\| ≤ 1` | worst over **2,200,000** draws (200k log-uniform on ±[1e-10, 1e10] × θ∈[−1000,1000], plus 2M pinned at `m=1`) | `1.0` exactly, `count(\|a\|>1)=0`, all finite | `1.0` exactly, `count(\|a\|>1)=0`, all finite | **HOLDS, identical** |
| **1** endpoints | `±inf`, `±1e308`, R1's eight `â_max` incl. `285.07` | `≤ 1.0`, finite | `≤ 1.0`, finite | **HOLDS** |
| **1** plant | `cap=False` must reach `285.0719` | `285.0719` | `285.0719` | fires on both |
| **2** band modulus | `\|Π e^{iθ}\|` over 1e4 phases | `prefix 1.0`, `max_abs_dev 1.1102230246251565e-16` | `prefix 0.9999999999999999`, `max_abs_dev 1.1102230246251565e-16` | **HOLDS** (`abs=1e-12`) |
| **2** `n_above_one` | the `≤ 1` bound | `0` of 10000 | `0` of 10000 | **HOLDS, identical** |
| **2** `n_exactly_one` | census of ulp-exact prefixes | `9767` | **`7713`** | **DIFFERS — §4.1** |
| **2** route gap | prefix vs cumulative product | `1.5654144647214707e-14` | `1.5654144647214707e-14` | identical |
| **2** plant | `m = 0.9` must collapse | `prefix 0.0` | `prefix 0.0` | fires on both |
| **3** parity | `torch.equal(parity, parity_sign_mask)` at `s=64` | **True, 0 / 4096** | **True, 0 / 4096** | **HOLDS, identical** |
| **3** winding integrality | residual | `2.1316282072803006e-14` | `1.0658141036401503e-14` | holds; cuda **tighter** |
| **3** continuous route | `exp(i(Φᵢ−Φⱼ))` vs mask | `6.762526253839743e-14` | `3.2341817148948714e-14` | holds; cuda tighter |
| **3** plant | `θ ∈ {0, π/2}` breaks it | residual `>0.4`, disagree `>0.1` | same | fires on both |
| **4** identity, operator | `torch.equal(op.real, lm.Attention("softmax_x").operator)` | **True (bitwise)** | **True (bitwise)** | **HOLDS** |
| **4** identity, imag | `torch.equal(op.imag, 0)` | True | True | **HOLDS** |
| **4** identity, `op.real @ v` | bitwise | True | True | **HOLDS** |
| **4** on the shipped `ArmPhase` | same, through the module | True | True | **HOLDS** |
| **4** complex-gemm gap | the ulp the bind does **not** reach | `1.1102230246251565e-16` | **`0.0`** | **DIFFERS — §4.2** |
| **4** plant | each of 3 heads moves the operator `>1e-3` | fires | fires | fires on both |
| **5** label bind, `s=8` | vs `1e-6` bar | **`9.155133597044475e-16`** | **`9.155133597044475e-16`** | **HOLDS, BITWISE THE PUBLISHED NUMBER** |
| **5** label bind, `s=64` | vs `1e-6` bar | `5.2510145522368515e-15` | `2.6749593496690556e-15` | holds; cuda tighter |
| **5** normalizer `Z=1` | drift | `4.440892098500626e-16` | `5.551115123125783e-16` | holds (`<1e-15`) |
| **5** band, `\|a\|=1` | must FAIL | `nan` | `nan` | the published cost, on both |
| **5** plants ×5 | all `> 1e-6` | `0.4239 / 0.8389 / 0.8893 / 0.9332 / 1.0` | `0.4239 / 0.8389 / 0.8893 / 0.9332 / 1.0` | fire on both |

**Every bind holds on both devices. Two published *numbers* differ, and neither
is a bind.**

#### 4.1 `n_exactly_one`: `9767` on cpu, `7713` on cuda

The bind is the four assertions above it — `prefix ≈ 1` to `1e-12`,
`product ≈ 1` to `1e-12`, `max_abs_dev < 1e-12`, `n_above_one == 0` — and all
four hold unchanged on both devices. `n_exactly_one` is a **census of how many of
the 10,000 prefixes land on `1.0` with no ulp of error at all**, which is a
property of the summation order inside `cumsum`: sequential on cpu, a parallel
scan on cuda. The extra 2,054 prefixes are still misses **downward**
(`n_above_one == 0` on both), so the `≤ 1` bound the bind actually claims is
untouched.

Pinned **per device, exact on both** (`9767` / `7713`), not widened to a range: a
range would stop being able to say which device moved.

#### 4.2 the complex-gemm gap: `1.1102230246251565e-16` on cpu, `0.0` on cuda

`V15_ARM_PHASE.md` (d) names this number precisely so nobody quotes "bitwise" for
the wrong tensor: the *operator* is bitwise, the *read-out* contracts it with a
complex gemm and is not. On cuda, cuBLAS returns the read-out **bitwise** and the
gap does not open at all.

**This is the device being stricter than the published claim, so nothing was
relaxed.** The test now asserts two things: the device-independent
`gap <= 1.1102230246251565e-16` (the published figure is a ceiling neither device
may exceed), and the exact per-device value — `0.0` on cuda. Admitting cuda by
widening the tolerance would have thrown away a fact.

### The shipped module

`ArmPhase(...).to("cuda")` constructs, runs `forward`, and returns `[n]`.
`identity_heads()` still yields exactly `m=1, θ=0, s=0` on device and the parity
bind survives the whole forward pass.

**`ceq/arm_phase.py` is still not wired into `scripts/v15_r1.py`.** That runner's
`make_arm` knows `arm_pl` and `m3_capability.Arm` only, and adding a third arm is
a construction, not device work. Naming it here because
`CEQ_V16_CONTRACT.md` names `arm_phase` for R1′ and this node did not close that
gap.

---

## 5. THE SUITE, RUN TWICE

`tests/arm_phase/test_arm_phase.py` gained `DEV`, read from
`ARM_PHASE_DEVICE` (default `cpu`), and every tensor source routes through it.
An env var and not a `params=` fixture on purpose: parametrising would rename
every test id and double the collected count.

| run | command | result |
|---|---|---|
| CPU | `python -m pytest tests/arm_phase -q` | **30 passed**, 1.76 s |
| CUDA | `ARM_PHASE_DEVICE=cuda python -m pytest tests/arm_phase -q` | **30 passed**, 4.15 s |

Same 30 ids, same count, diffable line for line. Two assertions added that are
strictly stronger than what was there:

- BIND 5 now asserts `residual == 9.155133597044475e-16` **exactly** on both
  devices (it was `< 1e-14`), and `cell["device"] == DEV.type`.
- BIND 4 now asserts the gap `<= 1.1102230246251565e-16` device-independently in
  addition to the exact per-device value.

`tests/arm_pl` (**not edited**, outside write scope): **17 passed**, and
`tests/arm_pl` + `tests/test_zero_step_gate.py`: **21 passed**. Both
confirm the `device` field change moved no cpu hash.

The zero-step gate-`R²` control reads **identically on both devices**, per seed:
`[0.196315, 0.146722, 0.246818, 0.594783, 0.086057, 0.322347, 0.095066, 0.64483]`,
mean `0.291617`. That is construct-then-move working end to end.

---

## 6. SILENT DOWNCAST — WHAT WAS LOOKED FOR AND WHAT WAS FOUND

### No dtype is silently narrowed on the bind path

Measured on cuda, op by op, over the ops `arm_phase` uses:

| op | returns | |
|---|---|---|
| `complex128 @ complex128` | `torch.complex128` | preserved |
| `cumsum(complex128)` | `torch.complex128` | preserved |
| `polar(float64, float64)` | `torch.complex128` | preserved |
| `angle(complex128)` | `torch.float64` | preserved |
| `exp(complex128)` | `torch.complex128` | preserved |
| `softmax(float64)` | `torch.float64` | preserved |
| `log(float64)` | `torch.float64` | preserved |

`complex128` matmul, cuda against a host float64 reference: `max|Δ| =
1.778576e-14` on `|ref|max = 23.71`, i.e. **`7.5e-16` relative** — float64
rounding, not a downcast. Flipping `allow_tf32` does not move it by one bit.

### The downcast that is there, and it is not in the bind path

**`ArmPhase`'s trained forward is `complex64`, not `complex128`.** `nn.Linear`
builds at the default `float32`, `_ctype(float32) = complex64`, and
`ArmPhase._operator` returns `torch.complex64`. `DTYPE`/`CDTYPE` —
`float64`/`complex128` — govern `label_cell` and the five binds, **not** the
module a training run would hold. Confirmed by construction on cuda:

```
x torch.float32 -> heads torch.float32 -> _operator torch.complex64
label_cell / the five binds: torch.complex128
```

That is a real gap between what is certified and what would be trained, and it
exists on cpu too — it is not a device finding. It becomes one here:

**TF32 propagates into `complex64` matmul on this card, and the effect is 3
orders of magnitude.**

| | `max|cuda − host|` | relative to `|ref|max = 23.71` |
|---|---|---|
| `complex64`, `allow_tf32=False` (the default) | `7.674894e-06` | `3.237e-07` |
| `complex64`, `allow_tf32=True` | **`7.012327e-03`** | **`2.958e-04`** |
| `complex128`, either setting | `1.778576e-14` | `7.502e-16` |

`torch.backends.cuda.matmul.allow_tf32` is **`False`** by default in
`torch 2.5.1`, so this does not bite today and nothing in the tree sets it. But
it is one global flag — the single most common "make the GPU run faster" edit —
away from running ARM PHASE's complex operator at roughly an 11-bit mantissa,
**silently**, with no dtype anywhere in the record changing. `identity_manifest`
has no field for it. A complex arm quietly running at reduced precision would
invalidate every identity bind it passes, and this is the mechanism by which
that would happen.

`torch.backends.cudnn.allow_tf32` is `True` by default; it governs cudnn
convolutions, which neither arm uses.

### The determinism blocker, measured

The LEAD CAVEAT's conflict, in full. Under `use_deterministic_algorithms(True)`
on cuda:

| callable | result |
|---|---|
| `torch.cumsum` (float64) | **RAISES** `cumsum_cuda_kernel does not have a deterministic implementation` |
| `torch.cumsum` (int64) | OK |
| `torch.cumprod` (float64) | OK |
| `arm_phase.scan_phase` / `key_bias` / `phase_factor` / `normalizer` / `winding` / `winding_matrix` / `operator` | **RAISE** |
| `arm_phase.magnitude` / `gate` / `parity_sign_mask` | OK |
| `arm_pl.scan`, `arm_pl.operator` | **RAISE** |
| `torch.softmax` (float64), `complex128` matmul, `torch.linalg.lstsq` | OK |

`cumsum_cuda_kernel` is the **only** nondeterministic op either arm touches —
confirmed by a `warn_only=True` sweep of a full `label_cell`, which names
exactly one op.

**Is the refusal guarding a real hazard at these shapes?** Measured, 20 repeats
in one process, same input:

| | max `|repeat − first|` over 20 |
|---|---|
| `cumsum` float64, `n = 64` | **`0.0`** |
| `cumsum` float64, `n = 10 000` | **`0.0`** |
| `cumsum` float64, `n = 1 000 000` | `6.821210263296962e-13` |
| `arm_phase.label_cell` residual, 20 repeats | **bitwise identical** |
| `arm_phase.band_modulus` prefix route, 10 repeats | **bitwise identical** |

So the kernel *is* genuinely nondeterministic — at `n = 1e6` it drifts run to
run — but at the arm's own scan length (`s = 64`, and `1e4` for the band probe)
it was bitwise reproducible over every repeat taken. **That is evidence, not a
licence**: 20 repeats on one card in one process is exactly the shape of claim
`V16_BAR_RECERT.md`'s own LIMITS flag as insufficient, and `1e6` shows the
hazard is real where the lane is long enough. `it11_verdict.by_seed`'s rule is
`1e-9`, which `6.82e-13` sits inside; that is a coincidence of scale, not a
guarantee.

Nothing here sets or clears the flag. `scripts/v15_r1.py` now journals
`deterministic_algorithms` in its header so the regime is on the record.

---

## 7. WHAT CHANGED, BY FILE

`git diff --stat`, this node's files only:

```
ceq/arm_phase.py                  |  55 +++++++++---
ceq/arm_pl.py                     |  27 +++++--
scripts/v15_r1.py                 | 150 ++++++++++++++++++++++++++---------
tests/arm_phase/test_arm_phase.py | 143 ++++++++++++++++++++++++---------
```

Executable diff, comments and docstrings excluded: four literals, one deleted
abort block, eleven threaded arguments, three `.cpu()` insertions, two
`device=` parameters, one `ones` → `ones_like`, and the mechanical `DEV` routing
in the test file. Everything else is the reasoning, at the point of use.

Untouched, as required: `scale/**`, `lean/**`, `ceq/beds/**`, `ceq/certs/**`,
`ceq/x35*/**`, `MISTAKES.md`, `tests/arm_pl/**`. No git command that writes was
run.

---

## 8. THE DECISION

| | status |
|---|---|
| **F1** — the four literals | **CLOSED.** Sites 309 / 430 / 441 / 492 record `a.device`. Guard demonstrated firing on a real mixed pool, and demonstrated *not* firing on either single-device control. |
| **F4** — the manifest literals | **CLOSED.** `arm_phase:415`, `arm_pl:255`. Two same-cell manifests, two devices, two hashes. No published cpu hash moved. |
| **F-ARM** — `arm_phase` on device | **RUNS.** All five binds hold on both devices; suite 30/30 on each. Two published numbers are device-dependent and both are named, pinned exactly per device, and neither is a bind. |
| complex downcast | **NONE on the `complex128` bind path.** The shipped `ArmPhase` module is `complex64`, and `allow_tf32=True` moves its product by `2.958e-04` relative — silent, unrecorded, one flag away. |
| **R1′ on cuda** | **STILL BLOCKED**, for a new reason: `cumsum_cuda_kernel` has no deterministic implementation, so the arm cannot execute in the regime the bar was certified in. |

### What the next node inherits

1. **Decide the determinism trade.** Either the arm runs with
   `use_deterministic_algorithms` off (and the bar is then read outside its
   certified regime), or with `warn_only=True` (same arithmetic, the refusal
   demoted to a warning, which is a decision that should be written down), or
   `cumsum` is replaced by a deterministic construction (`cumprod` is
   deterministic on cuda and `scan_phase`'s magnitude half is a `cumsum` of
   logs — a change of construction, out of scope here and probably a v16 first-law
   violation). §6's repeat measurements are what that decision should be made
   against, and they are 20 repeats on one card in one process.
2. **`arm_phase` is not reachable from `scripts/v15_r1.py`.** `make_arm` knows
   two arms; the contract names a third.
3. **`allow_tf32` has no manifest field.** A `complex64` arm plus a global flag
   is an unrecorded precision regime. `identity_manifest` is not this node's
   file.
4. **F2, F3, F5–F8 of `V16_BAR_RECERT.md` are untouched** and were out of scope.
   F2 (`THREAD_FLOOR` on cuda) is still the next `V-22` to land.

---

## LIMITS

- **Nothing was trained.** Every runner invocation is `--steps 0` at
  `n_train = n_eval = 512`, `seeds 0 1`. The `eval_nrmse` values in §2 are
  0-step readings and are **not** capability readings; they are there to show
  the guard cannot separate the two devices by value. R1′'s real shapes
  (`n_train=2048, n_eval=4096, 8 seeds, 150 steps`) were not run.
- **The device-comparability of *trained* `eval_nrmse` is still not
  established**, exactly as `V16_BAR_RECERT.md` §9 left it. This node removes
  the path by which a cuda reading could meet a cpu one unseen; it does not make
  them comparable, and the no-pooling rule is still what the argument rests on.
- **The bind side-by-side table is one seed** (`SEED = 15`, the fork probe's
  draw) at `s ∈ {8, 64}`, which is the shape `V15_ARM_PHASE.md` published at. No
  seed sweep of the label bind was taken on either device.
- **`n_exactly_one = 7713` is one card, one torch build.** It is a property of
  cuda's scan implementation and there is no reason to expect it stable across
  cards or torch versions the way `9767` is stable on cpu. Pinned so a change
  has to come to the test and say why, not because it is portable.
- **Run-to-run cuda determinism (§6) was measured in one process**, 20 repeats.
  Two separate processes were not compared. Same limit
  `V16_BAR_RECERT.md` §6.2 records for its own measurement.
- **The `t="bind"` rows now carry `device` but are not passed to
  `refuse_cross_device_pool`** — only the cell rows are, at line 469, as before.
  The field is there so the row is self-describing, not because a guard reads it.
- **`house-events.jsonl` shows as modified in `git status`** and was not touched
  by this node; a concurrent node or the harness is writing it.
- **`allow_tf32`'s effect was measured on one `64×64` `complex64` matmul**, not
  on `ArmPhase`'s forward at R1′'s shapes. The relative figure `2.958e-04` is a
  size, not the arm's error budget.
