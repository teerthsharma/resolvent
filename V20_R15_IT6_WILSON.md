# V20 R15 it.6 — WILSON

Scope: the six questions the round generated after `V20_R15_IT1_WILSON.md`. No
stance, no recommendation, no finding filed. Every claim carries the file, line,
command or output it rests on. Where a thing could not be verified it is listed
under **UNVERIFIED** at the end and nowhere else.

Box, for every measured cell below: `torch 2.5.1+cu121`, `python 3.11.9`,
`NVIDIA GeForce RTX 4060 Laptop GPU`, compute capability `(8, 9)`,
`torch.cuda.is_available() == True`, Windows 11.

The Inspector's premise checks out. `V20_R15_IT1_WILSON.md` contains **0**
occurrences of `warn_only`, `use_deterministic`, `_0step`,
`frac_gate_annihilated` and `n_zero_gates`, and 2 of `arm_phase` — at
`V20_R15_IT1_WILSON.md:31` (a module inventory row) and `:76` (a wiring row).
It stated nothing on any of the six questions below, in either direction.

Nurses: **none dispatched.** Every fetch this scope needed was a sub-second local
`grep`/`sed`/`python` on this box, and the standard set at it.1 — verbatim output,
no summary — is met by reading that output directly. Recorded so the absence is a
decision with a reason rather than an omission.

---

## 1. THE DETERMINISM REGIME, MEASURED

### 1.1 What was run

Each of the ten cells ran in its **own fresh subprocess** —
`torch.use_deterministic_algorithms` is a global, so a strict cell and a warn cell
in one process would not be two independent readings.
`CUBLAS_WORKSPACE_CONFIG=:4096:8` was exported before CUDA initialised, matching
`tests/gate0/conftest.py:26` and the regime the retake header journals (§2.1).
Device `cuda`, dtype `float64`, `S=8, D=16, N=4`.

Call forms used, each the shipped entry point:

- `arm_phase.operator(q, k, u, theta, s)` — `ceq/arm_phase.py:158`
- `arm_phase.scan_phase(magnitude(u), theta)` — `ceq/arm_phase.py:107`
- `arm_smprime.operator(q, k, u, theta)` — `ceq/arm_smprime.py:234`
- `arm_smprime.path_product(gate(magnitude(u), theta))` — `ceq/arm_smprime.py:144`
- `arm_smprime` **backward**: `ArmSMPrime(8, d_model=16).to(cuda).to(float64)`,
  then `m(x).sum().backward()` — `ceq/arm_smprime.py:497,572`

### 1.2 The ten cells

| target | `use_deterministic_algorithms(True)` strict | `warn_only=True` |
|---|---|---|
| `arm_phase.operator` | **RAISES** | **OK** (+UserWarning) |
| `arm_phase.scan_phase` | **RAISES** | **OK** (+UserWarning) |
| `arm_smprime.operator` | **OK**, no warning | **OK**, no warning |
| `arm_smprime.path_product` | **OK**, no warning | **OK**, no warning |
| `arm_smprime` **backward** | **RAISES** | **OK** (+UserWarning) |

Verbatim exception, byte-identical in all three RAISES cells:

```
RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application. You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.
```

Verbatim warning, byte-identical in all three warn-mode cells that warn:

```
UserWarning: cumsum_cuda_kernel does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\Context.cpp:95.)
```

The two `arm_smprime` **forward** cells emitted **no warning at all** under
`warn_only=True`. The absence of a warning is itself the reading: no
non-deterministic kernel was reached on that path.

### 1.3 The cell the round's citation does not contain

`arm_smprime`'s **backward** RAISES under strict while both of its measured
forward entry points are OK. Isolated to a single call with no module, no
optimizer and no softmax in the graph — `path_product` alone, then `.backward()`:

```
RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation, ...
path_product BACKWARD strict: RAISES
```

with the failing frame `g.abs().sum().backward()` over
`arm_smprime.path_product(a)`. The `torch.cumprod` at `ceq/arm_smprime.py:170`
has a deterministic CUDA forward and a backward that reaches `cumsum_cuda_kernel`.

`V16_ARM_SMPRIME.md:576-580` is a **forward-only** table:

```
  arm_smprime.operator        under use_deterministic_algorithms(True) on cuda: OK
  arm_smprime.path_product    under use_deterministic_algorithms(True) on cuda: OK
  arm_phase.operator          under use_deterministic_algorithms(True) on cuda: RAISES
      RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation
  arm_phase.scan_phase        under use_deterministic_algorithms(True) on cuda: RAISES
```

`V16_ARM_SMPRIME.md:582` reads `**cumprod has a deterministic CUDA kernel; cumsum does not.**`
That sentence is true of the forward and false of the backward, and the table
above it measures only the forward.

The round's own producer already says so in prose. `scripts/v15_r1.py:87-90`,
verbatim:

```
RULING 1 decides it. STRICT mode (`warn_only=False`) is NOT EXECUTABLE on a
cell this file takes: the cell trains for `--steps` gradient steps and the
backward of `cumprod` reaches `cumsum_cuda_kernel`, which has no deterministic
implementation in torch 2.5.1 (`COSTS.md` 1.6; `V17_R4_RETAKE_PRICE.md` 2.1
```

and again at `scripts/v15_r1.py:570-572`:

```
    #: whole run's flag; strict mode is not executable on a cell that trains
    #: (`cumsum_cuda_kernel` in the backward of `cumprod`), so this is the only
    #: one of the two ruled regimes an R1/R1' cell can be taken under. Set here
```

So the two offices' conclusion — that `arm_phase`'s documented RAISE does not bind
a `warn_only` round — reproduces here **for `arm_phase`'s forward**. The same
measurement shows `arm_smprime` is not exempt from the strict RAISE either; it is
exempt only in the forward.

### 1.4 Run-to-run repeatability actually attained under the round's regime

Measured, `warn_only=True`, cuda, float64, 5 repeats in one process, fresh model
at the same seed each repeat:

```
ArmSMPrime warn_only cuda float64, 5 repeats same process: forward BITWISE=True  backward BITWISE=True  max|grad delta|=0.000e+00
```

```
grad NaN count in repeat 0: 4803 / 4820
forward y repeat 0: nan
ArmPhase warn_only cuda: identical NaN MASK across 5 repeats = True
ArmPhase warn_only cuda: non-NaN entries BITWISE identical across 5 repeats = True
```

Two separate readings there.

(a) Both arms were bitwise repeatable in-process under `warn_only` on this box on
this run. A first pass reported `torch.equal == False` for `ArmPhase`; that was
NaN, not divergence — `torch.equal` is `False` whenever a NaN is present — and the
second listing disambiguates it by comparing the NaN mask and the non-NaN entries
separately.

(b) An **as-constructed** `ArmPhase` on cuda/float64 produced a **NaN forward** and
`4803 / 4820` NaN gradient entries. `ceq/arm_phase.py:121,128` take
`torch.log(m)` where `m = clamp(u, 0, 1)` (`ceq/arm_phase.py:96`), and an untrained
magnitude head reaches `0`. `scripts/v16_device_probe.py:975` names the same site:
`` WHY IT CAN FAIL. `ceq/arm_phase.py:121,128` take `log(m)` with ``.

Reading (a) is one process, one box, one run. It is not a statement about
across-process or across-box reproducibility, which was not measured.

### 1.5 The one site that sets strict unconditionally

`V16_ARM_SMPRIME.md:571-572` names `scale/r10_capacity_sweep.py::main()` as
setting strict `use_deterministic_algorithms(True)` unconditionally on its cuda
path. That is correct — `scale/r10_capacity_sweep.py:251-253`:

```python
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
```

The comment above it, `scale/r10_capacity_sweep.py:242-246`, records what the
decision to set it unconditionally was priced on:

```
        #: `torch.use_deterministic_algorithms(True)` cost
        #: 0.921x -- i.e. no slowdown, within run-to-run noise -- over 20 timed
        #: steps of this exact Arm/shape (s=64, n=2048, softmax) on cuda, so it
        #: is set unconditionally rather than left off to save a cost that does
        #: not exist.
```

The shape it was priced on is `softmax`. A `softmax` `Arm` reaches neither
`cumsum` nor `cumprod`, so it is the one arm of the three that cannot raise in
either direction under either regime. Stated as what the comment says, with no
consequence drawn.

---

## 2. WHAT `results/v17k_r4_retake.jsonl` CONTAINS

### 2.1 Header, verbatim, line 1

```
{"t": "header", "tag": "v17k_r4_retake", "task": "e3_t2", "t_star": 2, "s": 64, "d": 24, "n_train": 2048, "n_eval": 4096, "steps": 150, "seeds": [0, 1, 2, 3, 4, 5, 6, 7], "arms": ["arm_pl", "arm_smprime", "softmax"], "lr": 0.02, "d_model": 16, "device": "cuda", "deterministic_algorithms": true, "deterministic_warn_only": true, "cublas_workspace_config": ":4096:8", "threads": 8, "torch": "2.5.1+cu121", "floor_1": 0.7071067811865476, "instrument_hash": "5d41a63d57671725384b33249693d884ec17d0dcca11be7b3e58d6619bb9a309", "when": "2026-08-31 23:08:44"}
```

The regime fields are `"deterministic_algorithms": true`,
`"deterministic_warn_only": true`, `"cublas_workspace_config": ":4096:8"`. Set at
`scripts/v15_r1.py:575`.

### 2.2 Record counts by `t`

`wc -l results/v17k_r4_retake.jsonl` reads `424`. All 424 lines parse as JSON and
all 424 carry a `t` field.

| `t` | count |
|---|---|
| `agg` | 3 |
| `bar` | 1 |
| `bind` | 4 |
| `cell` | 24 |
| `contrast` | 2 |
| `header` | 1 |
| `identity` | 2 |
| `probe` | 2 |
| `trace` | 384 |
| `wall` | 1 |
| **total** | **424** |

`kind` over the 24 `t="cell"` records: `arm_pl` 8, `arm_smprime` 8, `softmax` 8.

### 2.3 The `t="cell"` column list

Union over the 24 cell records, first-seen order, **60 columns**:

```
t, kind, seed, steps, n_params, nrmse0_train, nrmse0_eval, red_ok, train_nrmse,
eval_nrmse, boot_lo, boot_hi, secs, conservation_drift,
conservation_drift_ex_row0, v_max, a_hat_max, a_hat_min, lambda_hat,
lambda_hat_live, frac_gate_annihilated, unit_root, dyn_range_bound,
z_winding_max, z_winding_residual, dist_to_skyline, dist_to_skyline_why,
sign_r2, sign_acc, sign_sst, c, gate_r2, loga_sst, sign_acc_0step,
gate_r2_0step, a_hat_max_0step, a_hat_min_0step, lambda_hat_0step,
lambda_hat_live_0step, frac_gate_annihilated_0step, unit_root_0step,
dyn_range_bound_0step, z_winding_max_0step, z_winding_residual_0step,
eval_h_hat, dist_to_floor, task, t_star, n_train, n_eval, s, d, d_model, device,
threads, torch_version, cell, floor_1, instrument_hash, manifest
```

**The 24 cell records do not share one schema.** The 8 `arm_pl` and 8
`arm_smprime` records carry all 60. Each of the 8 `softmax` records is **missing
17 columns**, the identical set on all eight:

```
sign_r2, sign_acc, sign_sst, c, gate_r2, loga_sst, sign_acc_0step,
gate_r2_0step, a_hat_max_0step, a_hat_min_0step, lambda_hat_0step,
lambda_hat_live_0step, frac_gate_annihilated_0step, unit_root_0step,
dyn_range_bound_0step, z_winding_max_0step, z_winding_residual_0step
```

No `softmax` cell carries any extra column. The cause is at
`scripts/v15_r1.py:771` — the whole probe-and-`_0step` block sits inside
`if kind in GATED_ARMS:`, and `GATED_ARMS = ("arm_pl", "arm_smprime")` at
`scripts/v15_r1.py:146`.

### 2.4 Per-seed values

The file's own JSON literals, unrounded.

**`arm_smprime`**

| seed | `eval_nrmse` | `frac_gate_annihilated` | `frac_gate_annihilated_0step` | `lambda_hat` | `unit_root` | `a_hat_min` | `a_hat_max` | `secs` |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.9260818361710341 | 0.5032958984375 | 0.5048828125 | -inf | True | 0.0 | 1.0 | 16.164 |
| 1 | 0.8812466551692475 | 0.5032958984375 | 0.611328125 | -inf | True | 0.0 | 1.0 | 16.032 |
| 2 | 0.20391993939877656 | 0.0 | 0.124755859375 | -0.795290470123291 | False | 0.340760201215744 | 0.5400443077087402 | 14.852 |
| 3 | 0.9162579895429029 | 0.4967041015625 | 0.44287109375 | -inf | False | 0.0 | 0.8454325795173645 | 15.977 |
| 4 | 0.9119259828412737 | 0.5032958984375 | 0.5157470703125 | -inf | True | 0.0 | 1.0 | 16.527 |
| 5 | 0.9181568234297154 | 0.5032958984375 | 0.5882568359375 | -inf | True | 0.0 | 1.0 | 16.464 |
| 6 | 0.8932391322242402 | 0.5032958984375 | 0.57958984375 | -inf | True | 0.0 | 1.0 | 16.66 |
| 7 | 0.9267169213804543 | 0.5032958984375 | 0.2005615234375 | -inf | True | 0.0 | 1.0 | 16.611 |

**`arm_pl`**

| seed | `eval_nrmse` | `frac_gate_annihilated` | `frac_gate_annihilated_0step` | `lambda_hat` | `unit_root` | `a_hat_min` | `a_hat_max` | `secs` |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6446726192039927 | 0.0 | 0.0 | -1.4324781894683838 | True | 0.043322015553712845 | 1.4104527235031128 | 1.884 |
| 1 | 0.6445174549187496 | 0.0 | 0.0 | -1.4365959167480469 | True | 0.04698067530989647 | 1.2868505716323853 | 1.739 |
| 2 | 1.1522795055459243 | 0.0 | 0.0 | 0.7584310173988342 | True | 0.3407682776451111 | 12.767516136169434 | 1.736 |
| 3 | 1.1133392329955414 | 0.0 | 0.0 | 0.1889575868844986 | True | 0.03181155025959015 | 49.6605224609375 | 1.916 |
| 4 | 0.6337391039935976 | 0.0 | 0.0 | -1.3844859600067139 | True | 0.0447322279214859 | 1.4536346197128296 | 1.759 |
| 5 | 0.6419986310848815 | 0.0 | 0.0 | -1.4528452157974243 | True | 0.03894248977303505 | 1.5051767826080322 | 1.745 |
| 6 | 0.6621282051474511 | 0.0 | 0.0 | -1.4714210033416748 | True | 0.046850644052028656 | 1.102945327758789 | 1.735 |
| 7 | 1.1489267727154717 | 0.0 | 0.0 | 1.2168586254119873 | True | 0.05253848060965538 | 116.00607299804688 | 1.726 |

**`softmax`**

| seed | `eval_nrmse` | `frac_gate_annihilated` | `frac_gate_annihilated_0step` | `lambda_hat` | `unit_root` | `a_hat_min` | `a_hat_max` | `secs` |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.9734002295313872 | 0.0 | **ABSENT** | 0.0 | True | 1.0 | 1.0 | 1.66 |
| 1 | 0.9387956332491751 | 0.0 | **ABSENT** | 0.0 | True | 1.0 | 1.0 | 1.665 |
| 2 | 0.9455472186895695 | 0.0 | **ABSENT** | 0.0 | True | 1.0 | 1.0 | 1.672 |
| 3 | 0.9515546327151698 | 0.0 | **ABSENT** | 0.0 | True | 1.0 | 1.0 | 1.663 |
| 4 | 0.9605147968181855 | 0.0 | **ABSENT** | 0.0 | True | 1.0 | 1.0 | 1.674 |
| 5 | 0.9456702325679535 | 0.0 | **ABSENT** | 0.0 | True | 1.0 | 1.0 | 1.749 |
| 6 | 0.960665082247913 | 0.0 | **ABSENT** | 0.0 | True | 1.0 | 1.0 | 1.671 |
| 7 | 0.9443373459209473 | 0.0 | **ABSENT** | 0.0 | True | 1.0 | 1.0 | 1.694 |

**ABSENT** means the key is not present in the record. It is not `null` and not
`0`; `json.loads(line)` has no such key.

No crossing count, rate, mean or any other derived statistic is computed here.

---

## 3. THE `_0step` COLUMNS

### 3.1 They exist, for two of three arms

`frac_gate_annihilated_0step` is present on all 16 gated cells and on none of the
8 `softmax` cells (§2.3, §2.4).

The eight `frac_gate_annihilated_0step` values `V20_R15_IT5_MARS.md:26-29` quotes
inside its RED, and the eight trained `frac_gate_annihilated` values beside them,
**reproduce digit-for-digit** from `results/v17k_r4_retake.jsonl` — compare
§2.4's `arm_smprime` table. The column MARS's STRIKE 8 reads exists and holds the
values MARS reports. Whether the `softmax` control's absence from that column
(§2.3) bears on the strike is not a question this office answers.

### 3.2 Where it is written

`scripts/v15_r1.py:808`, the single write site:

```python
                r.update({f"{k}_0step": v for k, v in
                          gate_columns(kind, m0, x_ev, live).items()})
```

`m0` is built six lines above, `scripts/v15_r1.py:800-802`:

```python
                torch.manual_seed(seed)
                m0 = make_arm(kind, S).to(dev)
                f0t, f0e = gate_features(m0, x_tr, live), gate_features(m0, x_ev, live)
```

The whole block is guarded by `scripts/v15_r1.py:771` — `if kind in GATED_ARMS:` —
with `GATED_ARMS = ("arm_pl", "arm_smprime")` at `scripts/v15_r1.py:146`.

### 3.3 What it is computed from

`gate_columns` is defined at `scripts/v15_r1.py:343`; the field itself is written
at `scripts/v15_r1.py:386`:

```python
               frac_gate_annihilated=float((~fin).double().mean()),
```

with `fin = torch.isfinite(lg)` at `scripts/v15_r1.py:380`, and `lg` set per arm at
`scripts/v15_r1.py:361-372`: for `arm_smprime`, `lg = torch.log(m)` where
`m, theta = arm_smprime.blend(*model.heads(x), model.g)` restricted to `live`; for
`arm_pl`, `lg = model.heads(x)[0][:, live]`, the head output itself.

**Measured before any optimizer step: yes.** `m0` is a **freshly constructed** arm
(`make_arm`, `scripts/v15_r1.py:179`) re-seeded to `seed` immediately before
construction, so its weights are the ones training started from. No optimizer
touches it and it never enters the training loop. The producer's own comment,
`scripts/v15_r1.py:794-799`, verbatim:

```
                #: read as a learned one -- the shape of MISTAKES.md V-10. The
                #: model is rebuilt under the same seed, so its weights are
                #: bitwise the ones training started from.
                #: AS CONSTRUCTED, not at the identity point. The control's job
                #: is to say what the probe reads off the weights TRAINING
                #: STARTED FROM, so `identity_point` must NOT be called here --
```

**Derived from another column at write time: no.** It is an independent forward
pass of `gate_columns` over `x_ev` on `m0`. It is not a function of
`frac_gate_annihilated`, and the two disagree numerically on 8 of the 16 gated
cells (§2.4 — e.g. `arm_smprime` seed 7: `0.5032958984375` against
`0.2005615234375`).

Two structural facts about the same column, stated because they bear on what it
can be compared against, and not as a finding:

- For `arm_pl`, `lg` is a raw `nn.Linear` output (`scripts/v15_r1.py:363`), finite
  except at overflow, so `frac_gate_annihilated` and its `_0step` twin are
  structurally `0.0` for that arm rather than a measured absence of annihilation.
  All 16 `arm_pl` values in §2.4 are `0.0`.
- For `softmax` the function **returns hardcoded constants and exits before the
  measurement** — `scripts/v15_r1.py:375-379`:

```python
        return dict(a_hat_max=1.0, a_hat_min=1.0, lambda_hat=0.0,
                    lambda_hat_live=0.0, frac_gate_annihilated=0.0,
                    unit_root=True, dyn_range_bound=math.inf,
                    z_winding_max=None, z_winding_residual=None)
```

  so every `softmax` entry in §2.4's `frac_gate_annihilated`, `lambda_hat`,
  `unit_root`, `a_hat_min` and `a_hat_max` columns is that literal, not a reading.

### 3.4 `n_zero_gates` — the claim verified

The claim that `n_zero_gates` is manufactured from `frac_gate_annihilated` at
`scripts/v15_r1.py:867-868` is **correct**. Verbatim, `scripts/v15_r1.py:867-868`:

```python
                            n_zero_gates=int(round(r["frac_gate_annihilated"]
                                                   * a.n_eval * len(live))))
```

It sits inside the `base.update(...)` beginning at `scripts/v15_r1.py:860` that
feeds `arm_smprime.cell_manifest` at `scripts/v15_r1.py:869`.

The retake's own records satisfy that arithmetic exactly, with no rounding
residue, on all eight `arm_smprime` cells — `a.n_eval * len(live) = 4096 * 2 = 8192`:

| seed | `frac_gate_annihilated` | `× 8192` | `manifest.smp_values.n_zero_gates` | exact integer |
|---|---|---|---|---|
| 0 | 0.5032958984375 | 4123.0 | 4123 | yes |
| 1 | 0.5032958984375 | 4123.0 | 4123 | yes |
| 2 | 0.0 | 0.0 | 0 | yes |
| 3 | 0.4967041015625 | 4069.0 | 4069 | yes |
| 4 | 0.5032958984375 | 4123.0 | 4123 | yes |
| 5 | 0.5032958984375 | 4123.0 | 4123 | yes |
| 6 | 0.5032958984375 | 4123.0 | 4123 | yes |
| 7 | 0.5032958984375 | 4123.0 | 4123 | yes |

So the published `n_zero_gates` carries no information the published
`frac_gate_annihilated` does not already carry.

An independent, directly measured `n_zero_gates` does exist in the arm module —
`ceq/arm_smprime.py:368`:

```python
        "n_zero_gates": int((live.abs() == 0).sum()),
```

— and it is **not** the one written into the retake's cell manifests. The manifest
field is declared in `SMP_FIELDS` at `ceq/arm_smprime.py:95`. A third site,
`scripts/v15_r1.py:538`, reads `rec["n_zero_gates"]` off a record rather than
computing it.

---

## 4. `arm_phase` IN THE TREE

### 4.1 Search across every result file, any key

```
$ find results -type f | wc -l
323
$ find results -type d
results
results/kaggle_v17k_output
results/m3_quintuple_v2_cuda_weights
results/m3_quintuple_v2_weights
results/paired
$ grep -rla "arm_phase" results/
results/k_cert_local.json
```

`grep -rla` searches every file including binaries and matches anywhere in the
line, so it is restricted neither to a `kind` key nor to any key at all.

**One hit in the whole of `results/`**, and it is a prose note, not an identity —
`results/k_cert_local.json:1198`:

```
    "note": "arm_smprime reduces with CUMPROD; cumsum is measured only as the contrast arm_phase/arm_pl hit",
```

No record anywhere in `results/` has `arm_phase` as the value of `kind`, or of any
other key. Extension census of the 323 files:
`165 .pt, 55 .txt, 51 .jsonl, 21 .md, 17 .log, 11 .json, 2 .sha256, 1 .lock`.

The 165 `.pt` checkpoints are covered by that grep, but a grep over a pickle is
weak evidence, so all 165 were **opened and their identity keys read** rather than
searched. `torch.load(..., weights_only=True)`, 165/165 readable, 0 failures:

| identity | value | count |
|---|---|---|
| `kind` | `'softmax'` | 158 |
| `arm` | `'pivot_signed'` | 2 |
| `arm` | `'pivot_unsigned'` | 2 |
| `config['arm']` | `'pivot_unsigned'` | 3 |
| **total** | | **165** |

**No checkpoint carries `arm_phase` under `kind`, `arm`, `config['arm']`, or any
other key.** The 158 that carry `kind` also carry a `state_dict` whose top-level
module names are `('mlp', 'readout', 'wk', 'wq')` in all 158 — no `m_head`,
`theta_head`, `s_head`, `beta`, `qk` or `g`, so none of them is a gated arm of any
kind, let alone an `ArmPhase` (whose constructor at `ceq/arm_phase.py:479-489`
would add `m_head`, `theta_head` and `s_head`). The 4 `results/paired/*.pt` carry
`arm` at the top level; the 3 `results/phaseD_weights_*.pt` carry it inside
`config`.

### 4.2 What `ceq/arm_phase.py` says about training at `:476`

Verbatim, `ceq/arm_phase.py:476`, inside the `ArmPhase` class docstring:

```
    NOT TRAINED HERE and not by this node (L-LEAN). No optimizer, no gradient.
```

The module docstring says it once more at `ceq/arm_phase.py:46`:

```
NOTHING HERE TRAINS (L-LEAN). No optimizer, no gradient step, no fit. Every
```

Both are scoped — `HERE`, `this node`, `this module`.

### 4.3 An optimizer path does exist for `arm_phase`

Not in `ceq/arm_phase.py`, and not in the round's producer. It exists in
`scripts/v16_device_probe.py`, which lists `arm_phase` in its own arm set at
`scripts/v16_device_probe.py:80` (`ARMS = ("softmax", "arm_pl", "arm_phase")`) and
runs 14 Adam steps on it. Verbatim, `scripts/v16_device_probe.py:886-903`:

```python
        def make():
            torch.manual_seed(0)
            return (arm_pl.ArmPL(64, d_model=D_MODEL) if kind == "arm_pl" else
                    ArmPhase(64, d_model=D_MODEL) if kind == "arm_phase" else
                    Arm(kind, 64)).to("cuda")
        try:
            m = make()
            o = torch.optim.Adam(m.parameters(), lr=LR)
            torch.manual_seed(1)
            x = torch.randn(2048, 64, D_MODEL, device="cuda")
            y = torch.randn(2048, device="cuda")
            per = []
            for i in range(14):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                o.zero_grad()
                torch.nn.functional.mse_loss(m(x), y).backward()
                o.step()
```

Every `ArmPhase(` construction outside `attic/` and `kaggle/`:

```
ceq/arm_phase.py:456                   class ArmPhase(nn.Module):
scripts/v16_device_probe.py:87         return ArmPhase(seq, d_model=D_MODEL)
scripts/v16_device_probe.py:889        ArmPhase(64, d_model=D_MODEL) if kind == "arm_phase" else
scripts/v16_device_probe.py:1082       m = ArmPhase(S, d_model=D_MODEL).identity_heads().to("cuda")
tests/arm_phase/test_arm_phase.py:308  arm = arm_phase.ArmPhase(s=S + 1, d_model=4, hidden=8).identity_heads().to(DEV)
tests/arm_phase/test_arm_phase.py:509  arm = arm_phase.ArmPhase(s, d_model=d_model).to(DEV)
```

The round's producer cannot build one. `scripts/v15_r1.py:179-184`:

```python
def make_arm(kind: str, s: int):
    if kind == "arm_pl":
        return arm_pl.ArmPL(s, d_model=D_MODEL)
    if kind == "arm_smprime":
        return arm_smprime.ArmSMPrime(s, d_model=D_MODEL)
    return Arm(kind, s)
```

There is no `arm_phase` branch; the string would fall through to `Arm(kind, s)`.
`scripts/v15_r1.py` imports the module at `:118` and uses exactly one thing from
it — `arm_phase.winding(theta)` at `:404`, as the winding instrument for
`arm_smprime`'s phase.

Files outside `attic/` and `kaggle/` that name `arm_phase` or `ArmPhase` at all:

```
ceq/arm_phase.py
ceq/arm_smprime.py
ceq/compat.py
scripts/k_cert.py
scripts/v15_r1.py
scripts/v16_device_probe.py
tests/arm_phase/test_arm_phase.py
tests/arm_smprime/test_arm_smprime.py
tests/gate0/test_g01_identity.py
tests/gate0/test_g11_label_cell.py
tests/mars_v20/test_the_three_arms_are_one_operator_at_beta_one.py
tests/mercury/test_v20_r15_it3_arena_price.py
tests/saturn/test_v20_r15_freeze_manifest.py
tests/saturn/test_v20_r15_wing_rubric.py
tests/saturn/test_v20_r15_wings_distinct.py
tests/saturn/test_v20_r15_wings_distinct_percell.py
```

---

## 5. THE SUITE, NOW

Command, from the repo root, at `HEAD 207e7b9` with the working tree as found:

```
python -m pytest -q -p no:cacheprovider
```

Nothing was fixed, edited, skipped, deselected or reverted. `-p no:cacheprovider`
was added for one reason only — so the run does not write a `.pytest_cache` into
the tree this office was told to leave alone. It deselects nothing.

### 5.1 Collection, which is now clean

```
$ python -m pytest -q --collect-only -p no:cacheprovider
3027 tests collected in 8.92s
```

**3027 tests, zero collection errors.** This is the reading `pytest.ini`'s
`norecursedirs` change (§5.4) produces. The journal's figure for the same command
at it.1 was `2895 tests` (`V20_R15_JOURNAL.md:193-194`); the tree has gained test
files since.

Slow-marked tests are **not** excluded and were not excluded here:

```
$ python -m pytest -q --collect-only -m slow -p no:cacheprovider
41/3027 tests collected (2986 deselected) in 8.82s
```

Collection by directory, largest first: `tests/loop` 541, `tests/chase` 489,
`tests/cameron` 454, `tests/gate0` 307, `tests/foreman` 297, `tests/jupiter` 125,
`tests/saturn` 97, `tests/mercury` 93, `tests/wilson` 70, `tests/arm_smprime` 52,
`tests/mars` 46, `tests/x35p` 44.

### 5.2 The tree moved under the run, and by how much

The run's process began at **`02:28:42`** — read off the OS, not inferred:

```
ProcessId : 20432
Start     : 02-09-2026 02:28:42
CmdLine   : ...python.exe -m pytest -q -p no:cacheprovider
```

Other offices were writing test files into the tree while it ran. **Four** it.6
test files were created after that instant and are therefore **not** in the run
below, while the `3027` figure in §5.1, measured later, does include them:

```
2026-09-02 02:30:38  tests/mercury/test_v20_r15_it6_seeds8_15.py
2026-09-02 02:34:55  tests/jupiter/test_v20_r15_it6_ldom_census.py
2026-09-02 02:36:16  tests/jupiter/test_v20_r15_it6_q1_exact_class.py
2026-09-02 02:38:04  tests/jupiter/test_v20_r15_it6_q2_outside_bound.py
```

So the run's own collected count and the `3027` of §5.1 are **two different
scopes**, and the run's summary line is the authoritative one for the run.

`.pytest_cache/` exists in the tree but was not created by this run — mtime
`2026-08-24 20:12:55`, and it is ignored (`.gitignore:11`). Both commands above
ran with `-p no:cacheprovider`.

### 5.3 The run

PLACEHOLDER_SECTION_5

### 5.4 `git diff pytest.ini`, verbatim

```
warning: in the working copy of 'pytest.ini', LF will be replaced by CRLF the next time Git touches it
diff --git a/pytest.ini b/pytest.ini
index b2c3f74..63e84d9 100644
--- a/pytest.ini
+++ b/pytest.ini
@@ -8,4 +8,11 @@ markers =
 # collection, because their `from conftest import requires_triton` resolved through
 # tests/chase/conftest.py, which is KEEP and did not move with them.
 # An explicitly-named path still collects, which is what the attic-aware guards need.
-norecursedirs = attic *.egg .* _darcs build CVS dist node_modules venv {arch}
+# kaggle/ holds a REGENERATED snapshot of this repo (kaggle/snapshot/repo/), so every
+# test basename exists twice. pytest imports the snapshot copy first and then refuses
+# the real one with "import file mismatch", and because that is a COLLECTION error the
+# run aborts before a single test executes -- `pytest -q` at the root read
+# `201 errors during collection`, 0 tests run, while `pytest tests/x35p/test_source.py`
+# on its own read `15 passed`. A dead suite reports no failures, which is why nothing
+# caught it: green and uncollectable look identical from the summary line.
+norecursedirs = attic kaggle *.egg .* _darcs build CVS dist node_modules venv {arch}
```

The change is **uncommitted** — `git status --porcelain` reads `M pytest.ini`, and
`git log --oneline -- pytest.ini` has `b3f44ff` as its most recent commit, which
predates it.

**Owner, claimed in writing.** `V20_R15_JOURNAL.md:978-984`, verbatim:

```
**OWNERSHIP, CLAIMED.** VENUS flagged `pytest.ini` as modified by an unattributed
process and correctly refused to revert it. **It is this office's edit**, made at it.1:
`norecursedirs` gained `kaggle` because `kaggle/snapshot/repo/` shadowed every test
basename and `pytest -q` at the root read `201 errors during collection, 0 tests run`.
It is recorded in the it.2 entry; it was **not** recorded anywhere an agent starting
fresh would look. **An unattributed edit against citations frozen at HEAD `207e7b9` is a
P-6 drift risk and she was right to name it.**
```

Independent corroboration of the timestamp at `V20_R15_IT1_INSPECTOR.md:17`:
`` `pytest.ini` gained `kaggle` to `norecursedirs` at mtime **`Sep 2 01:03:51`**. ``

Nothing was fixed, edited or reverted by this office.

---

## 6. THE LOG'S SCHEMA — `house-events.jsonl`

**Every count below is a SNAPSHOT, and it has to be.** The file is under
concurrent append by other offices while this report is being written: it read
`11898` lines when the schema census was taken and `12123` lines a short time
later, and of the 85 lines that landed between this office's own two events,
`83` are from `cameron` and `1` from `MERCURY`. Any line count in this section is
therefore true of that snapshot and of no other moment.

Two of the snapshot's lines are this office's own — `11813`
(`{"t":"wilson_it6_start",...}`) and `11893` (`{"t":"wilson_measure",...}`). Both
carry a `t` field, as instructed, so neither enters the `t`-less count in §6.2.

### 6.1 Lines and parseability, at the snapshot

```
TOTAL LINES (splitlines): 11898
BLANK LINES: 0
PARSED JSON RECORDS: 11894
UNPARSEABLE LINES: 4
```

The four that do not parse, with the parser's own message:

```
line 1899  Invalid \escape: line 1 column 224 (char 223)
line 2937  Invalid \escape: line 1 column 161 (char 160)
line 2938  Invalid \escape: line 1 column 496 (char 495)
line 5871  Invalid \escape: line 1 column 458 (char 457)
```

Line 1899 begins `{"t":"finding","agent":"Wilson","text":"Lean: ...` and carries a
raw regex inside the JSON string; line 2937 begins
`{"t":"finding","agent":"Foreman","text":"it.0 AND THE BARRINGTON THEOREM ...` and
carries raw LaTeX. All four are LaTeX or regex backslashes written into a JSON
string without escaping. All four **do** carry a `t` field as raw text; they are
counted here as unparseable, not as `t`-less.

### 6.2 Records with no `t` field

**116** of the 11,894 parsed records have no `t` key.

### 6.3 The alternative keys used instead

| key used instead of `t` | count among the 116 | first occurrence |
|---|---|---|
| `kind` | 114 | line 399 |
| `event` | 1 | line 11744 |
| *(none of `kind`, `event`, `type`, `evt`, `name`, `tag`, `phase`, `stage`, `action`)* | 1 | line 11719 |

First-key histogram over the 116: `agent` 114, `ts` 2.

Example, line 399, first 320 characters — the `kind` form:

```
{"agent":"chase","kind":"dispatch","round":4,"task":"make ceq/hf publishable and hostile-reader-proof","scope":["ship the parity operator sgate in the package","three silent-failure classes fail loudly","one-command CPU smoke test","costs stated inside the package"],"constraints":["TDD, every claim RED first in tests/c
```

Example, line 11744, first 320 characters — the sole `event` record:

```
{"ts": "2026-09-02T01:54:40", "round": "v20_r15", "iteration": 3, "node": "SATURN", "event": "report", "report": "V20_R15_IT3_SATURN.md", "tests": {"new_node": "tests/saturn/test_v20_r15_wings_distinct_percell.py", "red": 1, "green": 12, "suite_tests_saturn": "71 passed, 8 failed"}, "findings": {"A_mars_strike5": "UPHE
```

Example, line 11719, first 320 characters — no `t`, and no alternative identity key
at all:

```
{"ts": "2026-09-02 01:37:18", "round": "v20_r15", "iteration": 2, "agent": "SATURN", "report": "V20_R15_IT2_SATURN.md", "nodes": ["tests/saturn/test_v20_r15_wings_distinct.py", "tests/saturn/test_v20_r15_wing_rubric.py"], "A_wings_distinct": {"tolerance": 1e-12, "scaled_from": 9.547918011776346e-15, "factor": 105, "red
```

Line 11744 also uses `node` where the rest of the file uses `agent`, and both 11719
and 11744 use `ts` where records elsewhere do not.

### 6.4 `status` versus `state`

Over all 11,894 parsed records:

| | count |
|---|---|
| carrying `status` | 352 |
| carrying `state` | 10,827 |
| carrying **both** | 10 |
| carrying **neither** | 725 |

Counts only. No consequence is drawn.

---

## UNVERIFIED

PLACEHOLDER_UNVERIFIED_RUN

**U-2. Whether the retake run of `2026-08-31 23:08:44` was itself reproducible.**
§1.4 measures five repeats **in one process on this box today** and finds them
bitwise identical. It does not measure across processes, across boxes, or on the
2026-08-31 driver/library state, and the retake produced no checkpoint to compare
against — `scripts/v15_r1.py` contains no `torch.save`. Nothing here says the
journalled cells reproduce.

**U-3. Whether `arm_phase`'s NaN forward (§1.4b) also occurs at the settings the
round would use.** Measured at `d_model=16`, `S=8`, `float64`, cuda, weights
as-constructed under `torch.manual_seed(0)`. Not measured at the retake's
`s=64, d_model=16` shape, on cpu, in float32, or after `identity_heads()`.

**U-4. Why `arm_smprime` seed 2 escapes.** Nothing in this report addresses it.
§2.4 reports the numbers and §3 reports how the columns are produced; no
mechanism, cause, basin claim, or comparison is offered, and the absence of a
`softmax` `_0step` control (§2.3) is reported as a schema fact only.

**U-5. What the 4 unparseable `house-events.jsonl` lines contain past the
truncation point.** They are not valid JSON, so their fields cannot be read as
data; §6.1 reports only the parser's message, the leading bytes, and the fact
that each carries a literal `"t":` in its raw text.

**U-6. Whether the `t`-less and `status`/`state` counts in §6 hold now.** They do
not; they were true of the snapshot named in §6's preamble. The file is under
concurrent append and had grown by 225 lines before this report was finished.

**U-7. Whether `arm_phase` appears in result files outside `results/`.** The
question named `results/`, and the search in §4.1 covers `results/` and nothing
else. `attic/` and `kaggle/` were excluded from the source greps in §4.2-4.3 and
were not searched for records.

**U-8. Nothing — this one was closed rather than left open.** Every cited file was
re-stat'd after the report was written. All source and data citations rest on files
untouched since before this iteration:

```
2026-08-31 23:11:31  results/v17k_r4_retake.jsonl
2026-08-31 23:08:14  scripts/v15_r1.py
2026-08-31 19:28:58  ceq/arm_phase.py
2026-08-31 21:51:21  ceq/arm_smprime.py
2026-08-31 19:11:24  scale/r10_capacity_sweep.py
2026-08-31 19:55:00  scripts/v16_device_probe.py
2026-08-31 20:44:16  V16_ARM_SMPRIME.md
2026-08-31 23:14:54  tests/gate0/conftest.py
2026-09-02 01:03:51  pytest.ini
```

`pytest.ini`'s mtime `2026-09-02 01:03:51` matches `V20_R15_IT1_INSPECTOR.md:17`'s
independently-reported `Sep 2 01:03:51` to the second. Two cited files did move
during this iteration — `V20_R15_JOURNAL.md` (`02:40:08`) and
`V20_R15_IT5_MARS.md` (`02:20:09`) — so both quotes were re-located after the
write: `OWNERSHIP, CLAIMED` is still at `V20_R15_JOURNAL.md:978`, and the
`2895 tests` figure still at `:194`. `house-events.jsonl` moved and is handled at
U-6.
