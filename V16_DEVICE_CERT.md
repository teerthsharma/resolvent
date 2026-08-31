# V16 NEPTUNE — THE DEVICE CERTIFICATE

**D-DEVICE**, `CEQ_V16_CONTRACT.md` PART V: *"re-certification: 637 CPU-h vs
47 GPU-h is a 13× decision; NEPTUNE prices both with the calibrated model and
the round runs on the certified device only."* Round 12, it.4.

Probe: `scripts/v16_device_probe.py`, output pasted verbatim in §7. Repo HEAD
at the run of record `e292469`; **`7833953` by the time this was signed** —
*"Make the device live in R1-prime's journals, and find determinism and the
arms are incompatible on CUDA"*, which is `V16_R1_DEVICE_READY.md` and is
§5.3's conflict arriving from the other side.

**This certificate was written against a moving tree, and that is stated rather
than hidden.** `scale/negation_scope.py` and `scale/r10_capacity_sweep.py`
changed *during* the probe's own development — `calibrate_bar` gained a
`device=` parameter between run 1 and run 2, and the `--device cuda` abort was
deleted between run 2 and run 3 — and `ceq/arm_phase.py`, `ceq/arm_pl.py`,
`scripts/v15_r1.py` and `tests/arm_phase/test_arm_phase.py` changed after it.
§5.0 reads their state at run time rather than asserting it. The
`ceq/arm_phase.py` edit is device plumbing for the identity probes and does
**not** touch `magnitude`, the prefix scan or `ArmPhase.forward`, so §3 and
§5.4 stand against the current tree; **verified by read-only diff, not
assumed.**

**Provenance tag on every number.** `[MEASURED]` = read off this box in this
run. `[MODULE]` = computed by a named function in `ceq/sizing.py`. `[FITTED]` =
projected from a law fitted here, with its R². `[INHERITED]` = carried from an
earlier round, cited by file. `[ASSUMED]` = a value the contract does not fix,
chosen here with a reason. **NOT MEASURED** where it could not be obtained.

**L-TIME.** No timing figure below is inherited. The four this certificate
would otherwise have carried — `secs ~ n^1.338`, the previous round's CPU/CUDA
exponents, ARM PL's `91.63 s`, and `V15_MERCURY_DEVICE.md`'s determinism cost
`0.921×` — are all re-measured in §7, and every one of them moved.

**L-LEAN / "nothing trains as a research reading".** Every allocation and
timing below runs on `torch.randn` or on a corpus draw used for its shape and
bytes only. No NRMSE is scored, no seed is pooled, no verdict is formed, no
cell of R1′, R2 or any bed is run. §5.4 takes gradient steps and reports
**only whether the gradient is finite**. §5 calls `calibrate_bar` for its wall
clock and deliberately does **not** print its five clause values, because
printing them would be performing the re-certification D-DEVICE assigns
elsewhere — `V16_BAR_RECERT.md`, which landed while this ran and did it.

---

## 0. LEAD CAVEAT

**The device is CUDA and the certificate is signed — but the two things that
could stop the round are not the device, and a reader who takes the CUDA
verdict and stops has taken the wrong half.**

1. **`n = 32768` does not fit with the phase-gate arm, on either device.**
   R2 is the deciding cell of the round and it is registered at an `n` the arm
   it must run cannot allocate. This is a memory fact, it is independent of
   every device question in this file, and moving to CPU does not solve it.
   §3.4.
2. **The phase arm cannot complete 40 gradient steps on either device**, and
   fails at step 0 from its own initialiser. §5.4. **Every hour priced in §6.2
   assumes an arm that completes its steps**; if it.5 ships the arm as it
   stands, none of those hours is spendable.

Both are somebody's task and neither is this node's to fix. The device
decision below is correct and is also, on its own, not sufficient for the round
to run.

---

## 0.1 THE VERDICT IN SEVEN LINES

1. **The device is CUDA** — NVIDIA RTX 4060 Laptop, sm_89, 7.996 GiB — signed
   in §6.3. **The re-certification the contract called a 13× decision cost
   0.77 seconds of compute and is DONE** (`V16_BAR_RECERT.md`); what remains
   conditional is not a measurement but the regime mismatch in line 5. §5.2.
2. **R1's 5.8× is diagnosed to a mechanism.** It is neither the arm nor the
   shape nor the thread count: it is **thread PLACEMENT on a hybrid CPU crossed
   with host CONTENTION**, which no single-factor sweep finds. Reproduced at
   2.0×–4.0× of a 5.2×–5.8× target across six measurements — the mechanism is
   established, the magnitude is not, and the *variance* is the finding. §2.
3. **The calibrated memory model holds for the real-valued arms and FAILS for
   the complex arm.** `measured/predicted = 0.950` unchanged for softmax and
   ARM PL across a 16× span; **1.842 for ARM PHASE**, i.e. the module
   **under-predicts**, which is the direction a sizing gate must never have.
   `C_OPERATOR` for the complex arm is **7.50 at 8 B/element**. §3.
4. **R2 at `n = 32768` DOES NOT FIT with the phase-gate arm** — 7.768 GiB of
   activations against 6.939 GiB free and 7.996 GiB *total*. It does not fit on
   CPU either. Four contract cells are affected; the last power of two the
   phase arm affords is **`n = 16384`**. §3.4.
5. **The bar was certified in a regime the arms cannot execute in.**
   `V16_BAR_RECERT.md` certified `calibrate_bar` under
   `use_deterministic_algorithms(True)`; `cumsum` has no deterministic CUDA
   kernel and the prefix scan *is* §S-M′, so every gated arm RAISES under that
   flag. Three exits priced in §5.3.1; **the recommendation is exit 2**
   (`warn_only=True` plus a claim with a measured boundary), because it is the
   only exit that leaves the bar's regime and the run's regime identical.
6. **The phase arm cannot complete 40 gradient steps on either device.**
   `log(clamp(u,0,1))` is `−inf` at the closed endpoint the X₃₆ delta chose on
   purpose; 50.08% of positions sit there at construction. Device-independent.
   **Every hour in §6.2 is conditional on this.** §5.4.
7. **The 40-iteration plan costs 68.35 GPU-hours** at the full 9600 ladder and
   **1.07 GPU-hours** at the 150-step floor, against **815.88** and **12.75**
   CPU-hours. **The decision is 11.94×, not 13.6×** — and at the pessimistic
   end it is **29.0×**, because the CPU lane's spread is 2.55× and the CUDA
   lane's is 1.05×. §6.2.

---

## 1. THE BOX

| quantity | value | how |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, sm_89, 24 MPs | `torch.cuda.get_device_properties(0)` `[MEASURED]` |
| VRAM total | 8,585,216,000 B = **7.996 GiB** | `torch.cuda.mem_get_info()[1]` `[MEASURED]` |
| VRAM free | 7,451,181,056 B = **6.939 GiB** | `torch.cuda.mem_get_info()[0]` `[MEASURED]` |
| host RAM | 15.71 GiB total; 8.1–9.2 GiB available across this run's samples | `psutil.virtual_memory()` `[MEASURED]` |
| CPU | **Intel i7-14700HX, 20 physical / 28 logical** — 8 P-cores with SMT (logical 0–15) + 12 E-cores (logical 16–27) | `Win32_Processor`, with the P/E split confirmed by the timing asymmetry in §2.3 `[MEASURED]` |
| torch | 2.5.1+cu121, Windows 11 | `[MEASURED]` |

**The hybrid CPU is not trivia; it is §2's whole answer.** The previous systems
node recorded "28 logical, torch default 20 threads" and stopped there. Twelve
of those twenty-eight logical processors are not the same kind of core as the
other sixteen, and **`torch.set_num_threads(k)` names a count, never a
placement.**

The GPU is the same RTX 4060 Laptop that `ceq/sizing.py`'s docstring names as
the box its constants were calibrated on, so §3 is a re-validation on the
calibration hardware at a different workload shape and — new this round — at a
different **dtype**.

---

## 2. R1's 5.8× — THE DIAGNOSIS

### 2.1 What was filed, and what it forbade

`V15_R1.md` §9 records ARM PL at **`91.63 s`** per 150 steps at `n = 2048` and
softmax at **`71.32 s`**, against `V15_NEPTUNE_SYSTEMS.md` §4.2's re-measured
**`12.2 s`** for the same operator. R1 stated the gap is "the box, not the
arm", left it **undiagnosed**, and forbade repricing from it. Both halves were
right; the second is why this node exists.

Both filed numbers reproduce exactly from the journal, so the target is not in
doubt: `results/v15_r1.jsonl`'s eight softmax `secs` average to `71.324` and
its eight ARM PL `secs` to `91.634`. `[MEASURED from the journal]`

### 2.2 Six candidates eliminated by measurement

Every factor was varied **one at a time**, at R1's own pin (`threads = 8`),
R1's own step count (150), R1's own statistic (`total/steps`, warm-up included,
which is what `scripts/v15_r1.py:train_one` records as `secs`), and R1's own
corpus (`make_equilibrium_batch`, `t* = 2`, seed 0).

| candidate | measured | verdict |
|---|---|---|
| **thread count** (R1 pinned 8; the previous node 12) | `threads=8 / threads=12` = **1.058** in the run of record, and **0.99–1.26 across eight runs** of this probe — the comparison is itself dominated by §2.3's noise and never approaches 5.8 | **eliminated** |
| **data source** (`torch.randn` vs the real corpus) | softmax `1.011×`, ARM PL `1.027×`, ARM PHASE `0.997×` | **eliminated** |
| **the arm** (ARM PL vs softmax, same corpus) | **`1.099×`** here, `1.06×–1.11×` across runs — against R1's own filed `1.285`; see §2.5 | **eliminated** |
| **the statistic** (median of timed steps vs `total/150`) | `0.0819` vs `0.0829` = `1.012×` | **eliminated** |
| **sustained load / thermal drift** | over a 150-step run the last steps are not slower than the first; and R1's own softmax block got *faster* with time, which is the wrong sign for thermal | **eliminated** |
| **dtype and shape** | identical by construction — same `Arm`, same `s = 64`, same `n = 2048`, fp32 | not a variable |

**The headline: R1's cell reproduces at 0.17× of its filed number.** Softmax
at `n = 2048`, `threads = 8`, 150 steps on the real corpus reads **12.4 s**
here against R1's **`71.32 s`**. Across eight runs of this probe the same cell
read **12.3–13.6 s**, so **R1 is 5.2×–5.8× a quiet lane on this box** — and
`5.8×` sits inside that band, which is why the brief's figure is the right one
to have chased. The previous node's `12.2 s` reproduces exactly.

### 2.3 The two candidates that survive, and why neither is enough alone

**Host contention.** Concurrent 1-thread torch processes on 2048×2048 matmuls —
operands too large for this box's L3, so each costs one saturated core *and* a
share of DRAM bandwidth, which is the axis that can reach a cell bandwidth-bound
on a materialised `[n,s,s]`. Measured spread over the sweep: **2.55× on CPU**.
Adding 6 GiB of touched ballast, to model the host-memory pressure
`V13_DAG_TASKLIST.md:105-110` warns about, moved it no further.

**Thread placement.** Pinned to E-cores only, the identical cell costs
**1.44×** the P-core lane at rest.

| lane | `k` contenders | s/step | ×150 | vs P-core |
|---|---|---|---|---|
| P-cores, SMT siblings | 0 | 0.0912 | 13.7 s | 1.00× |
| P-cores, one thread each | 0 | 0.0931 | 14.0 s | 1.02× |
| default, no pin | 0 | 0.0818 | 12.3 s | 0.90× |
| **E-cores only** | 0 | 0.1315 | 19.7 s | **1.44×** |
| default, no pin | 12 | 0.1142 | 17.1 s | 1.25× |
| **E-cores only** | **12** | **0.2701** | **40.5 s** | **2.96×** |

`[MEASURED]`. Neither factor reaches 5.2 alone. **Their cross does most of the
rest, and it is not stable:** the E-cores × contention lane read **2.96×**
here, and **2.00×, 2.18×, 2.75×, 2.93×** in four earlier runs of this same
probe, and **4.03×** in an independent standalone measurement. Six
measurements of one configuration spanning **2.0× to 4.0×** — and the fact
that one configuration spans a factor of two is not noise around the finding,
it **is** the finding.

### 2.4 One dated corroboration that R1's box was in fact shared

`results/v15_r1.jsonl` opens at `2026-08-31 17:25:46` and its `wall` record is
`1358.32 s`, so the run closed at `17:48:24`. Summing the journalled `secs`
plus per-cell overhead places softmax seed 3's window at
**`17:42:26`–`17:43:41`**.

`lean/.lake/build/lib/CEQ/V15Source.olean` and its five siblings have mtime
**`17:43:35`** — a `lake build` finishing *inside that window*. It belongs to
commit `9ab7c2f`, authored `17:44`. Three earlier `lake build`s land at
`16:27`, `16:48` and `17:09`. `[MEASURED from mtimes and read-only git log]`

The four softmax cells **before** that timestamp average **`76.31 s`**; the
four **after** average **`66.34 s`** — a **13.1% step down** at the moment a
competing compile completed.

This does not measure the whole gap. It establishes, with a date, that R1's box
was carrying other nodes' work — which is the mechanism class §2.3 reproduces,
and which the campaign's parallel-planet dispatch makes the normal condition
rather than an accident.

### 2.5 R1's own arm-cost ratio is confounded with run order

R1 ran all eight ARM PL cells first (`17:25:46`–`17:37:59`) and all eight
softmax cells second. Its `1.285×` "ARM PL costs more than softmax" therefore
compares two *time windows* as much as two arms — and the softmax window is the
later, quieter one. Measured here in one process at the same shape, the ratio
is **`1.099×`**. R1's sentence "the arm is not the slow part" survives; the
number attached to it does not.

### 2.6 The ruling

**Diagnosed to a mechanism: thread placement crossed with host contention, on a
hybrid CPU whose thread count does not determine its thread placement.**
Reproduced at **2.0×–4.0×** of a **5.2×–5.8×** target across six independent
measurements — 41–83% of the gap in log terms. **The magnitude is not pinned
and this node does not claim it is.** The exact machine state of
`17:25`–`17:48` on 2026-08-31 is not recoverable.

**Three consequences, in order of what they cost:**

1. **R1's timings remain void for pricing**, as R1 itself ruled. Nothing in §6
   descends from them.
2. **A CPU schedule for this round has a free parameter in it.** The same cell,
   same settings, same box, reads between **12.3 s** and **40.5 s** per 150
   steps depending on where Windows puts eight threads and what else is
   running — a **3.3×** band inside this run alone, and **5.8×** if R1's own
   cell is admitted as a sample. That band multiplies every CPU hour in §6.2.
3. **The CUDA lane does not have that parameter.** Under the identical host
   pressure sweep the CUDA cell's spread is **1.05×** against the CPU cell's
   **2.55×**. This is the *schedulability* argument for the device and it is
   independent of the speed argument: even if CUDA were merely as fast as CPU,
   it would still be the lane whose wall clock is a property of the work rather
   than of the box's mood. **It is also the argument R1's undiagnosed gap
   actually supports** — a gap that exists only on the host is a reason to
   leave the host.

**If the round runs on CPU anyway, one line fixes most of this:** pin CPU
affinity, not only the thread count, and journal the affinity beside `threads`.
`scripts/v15_r1.py` and `scale/r10_capacity_sweep.py:206` both pin
`torch.set_num_threads(a.threads)` and stop there. Under M-10's own logic —
thread count is part of a cell's identity because it names the reduction lane —
**core class is part of a cell's identity too**, and nothing in the repository
records it. That is a `MISTAKES.md` candidate this node cannot file, being
outside its write scope.

---

## 3. THE MEMORY MODEL AGAINST A COMPLEX ARM

### 3.1 The `n` sweep: the model holds where it held, and breaks where it is new

Peak CUDA bytes for one forward + backward above a baseline taken after the
model, the inputs and the mask cache exist. `pred` is
`sizing.activation_bytes(cfg, batch=n, arm="signed", dtype="fp32")` `[MODULE]`
— `arm="signed"` and not `"softmax"`, for the reason `V15_NEPTUNE_SYSTEMS.md`
§2.2 gives: in `ceq/sizing.py` "softmax" means the fused SDPA path that forms
no `[S,S]` tensor, and every arm here materialises one.

| arm | `n` = 512 / 2,048 / 8,192 | predicted GiB `[MODULE]` | measured GiB `[MEASURED]` | m/p |
|---|---|---|---|---|
| softmax | | 0.066 / 0.263 / 1.054 | 0.063 / 0.250 / 1.000 | **0.950 / 0.950 / 0.949** |
| arm_pl | | 0.066 / 0.263 / 1.054 | 0.063 / 0.250 / 1.000 | **0.950 / 0.950 / 0.949** |
| **arm_phase** | | 0.066 / 0.263 / 1.054 | **0.121 / 0.485 / 1.941** | **1.842 / 1.842 / 1.842** |

**VERDICT, two halves.**

**The model HOLDS for the real-valued arms, unchanged and conservatively.**
`0.950` at every `n`, identical to the ratio the previous round measured over
`n = 512…32768`. It over-predicts by 5%, which is the right direction for a
gate. **ARM PL, which did not exist when the model was last validated,
transfers exactly** — its operator is `float32` like softmax's and its
constants back-solve to the same values to three figures.

**The model FAILS for the complex arm, and it fails UNDER-predicting.**
`m/p = 1.842`, constant across the same 16× span. `ceq/sizing.py` has no
complex dtype mode — `DTYPE_MODES` offers `fp32` and `bf16_autocast` only — so
`activation_bytes` prices an 8-byte operator at 4 bytes and **no argument to
it fixes this.** Under-prediction is the direction a sizing gate must never
have, and §3.4 is what it costs.

**The extrapolation is licensed model-free.** Both terms are linear in `n` at
fixed `s`, so bytes-per-example measured at one `n` *is* bytes-per-example at
every `n`. Measured spread over `n = 512…8192`: **0.09%** for softmax and
ARM PL, **0.02%** for ARM PHASE. The previous round's independent check — that
softmax's per-example figure at small `n` reproduces at `n = 32768` — confirms
the same extrapolation one octave beyond anything measured here, and §3.3's two
probes land at **0.999** of prediction.

### 3.2 The constants, back-solved per arm

Two `s` points at fixed `n` (64 and 512), two equations, two unknowns,
`C_HEAD` held at the module value. `s` is the only axis on which the two terms
scale differently (`s` against `s²`), so it is the only one that separates
them. Bytes-per-element of the operator is **read off the live tensor**.

| arm | operator dtype | B/element | `C_RESIDUAL` | `C_OPERATOR` |
|---|---|---|---|---|
| softmax | `float32` | 4 | 17.92 | **3.499** |
| arm_pl | `float32` | 4 | 17.92 | **3.500** |
| **arm_phase** | **`complex64`** | **8** | 2.02 | **7.500** |
| *module* | — | *4* | *18.00* | *3.90* |

`[MEASURED]` for the three arm rows; `[MODULE]` for the last.

**`C_OPERATOR` for a complex arm is 7.50 at 8 bytes per element — the operator
term is 4.29× softmax's in bytes, not 2×.** The naive expectation, that complex
costs twice fp32 because `complex64` is twice `float32`, is wrong by a further
factor of 2.14: `ceq/arm_phase.py` builds a real causal softmax row *and* a
complex twist `exp(i(Φ_i − Φ_j))` *and* their product, so autograd retains
roughly twice as many `[n,s,s]` tensors as the single-operator arms. **This is
the number `V15_NEPTUNE_SYSTEMS.md` §8 filed as NOT MEASURED for a composed
arm.** It is measured now, for the phase arm, and the previous node's guess
that a composed arm "may retain 6–8 rather than 3.5" was right in kind and
almost exactly right in size.

**`C_RESIDUAL = 2.02` for the phase arm is NOT a residual measurement.** At
`s = 64` the operator term is 84% of this arm's total, so the two-point solve
has almost no leverage on the `s`-linear coefficient and dumps the model's
error into it. The honest reading is that **for the complex arm the two-term
decomposition is not identified at this shape**; §3.3 and §3.4 therefore use
measured bytes-per-example rather than the decomposition. Stated rather than
presented as a residual count.

### 3.3 R2 at `n = 32768`, predicted then probed

Predicted with each arm's own constants, checked against free VRAM, and
**attempted only where the prediction permits it.**

| arm | predicted activations | predicted working set | measured | m/p | verdict |
|---|---|---|---|---|---|
| softmax | 4.005 GiB | 4.130 GiB | **4.000 GiB** | 0.999 | **FITS** |
| arm_pl | 4.006 GiB | 4.131 GiB | **4.000 GiB** | 0.999 | **FITS** |
| **arm_phase** | **7.768 GiB** | **7.893 GiB** | — | — | **REFUSED** |

The phase arm's row was **not attempted**: 7.893 GiB exceeds 6.939 GiB free
*and* exceeds 7.996 GiB total, so no allocation was made against a prediction
that says it cannot succeed. The two measured rows land at **0.999** of
prediction — the previous round's `4.000 GiB` at this cell reproduces to the
byte, and the per-arm constants predict it 5% more tightly than the module's.

### 3.4 The cells that do not fit

Largest affordable `n` from measured bytes-per-example (the pessimistic end of
its own 0.09% spread) plus `x_train`, against free VRAM less a 0.200 GiB
eval-set reserve for `x_eval` at `n_eval = 4096` and its no-grad forward.

| arm | B/example | `n` @ 0% margin | @ 10% | **@ 20%** | last 2^k |
|---|---|---|---|---|---|
| softmax | 135,321 | 52,809 | 47,528 | **42,247** | 32,768 |
| arm_pl | 135,329 | 52,806 | 47,525 | **42,245** | 32,768 |
| **arm_phase** | **258,609** | **27,633** | 24,870 | **22,106** | **16,384** |

**`n = 32768` with the phase-gate arm does not fit at ANY margin, including
zero.** Four contract cells are affected, and they are not minor ones:

| iteration | node | `n` | the arm that does not fit |
|---|---|---|---|
| **it.10** | **R2** BED-M `t*=8` | 32,768 | the phase-gate arm — *the deciding cell of the round* |
| it.17 | R3(a) delay bed | 32,768 | the composed arm, if complex — **NOT MEASURED** |
| it.18 | R3(b) power-law bed | 32,768 | the composed arm, if complex — **NOT MEASURED** |
| it.25 | R5 BED-1 | 32,768 | the phase-gate arm |

**It does not fit on CPU either.** 7.893 GiB against a host that showed
8.0–10.0 GiB available across this run's own samples, on a 15.71 GiB machine —
and that is *before* the interpreter, torch's image and both corpora, which
`V15_R1.md` measured at 1.0023 GiB of process working set for a cell 16× smaller.
**The CPU route is not a way around this.**

**Three options, priced. The choice is not this node's.**

| # | option | memory | R2's CUDA cost, 9600 ladder | what it costs the round |
|---|---|---|---|---|
| 1 | **`n = 16384`**, the last power of two the phase arm affords | **3.946 GiB**, 43% headroom | **5.29 h** (phase 3.58 + softmax 1.71) | R2's registered `n` changes. `floor₁ = 0.9354` is a closed form in `t*` and does not move; halving the training set is a **capability** question this node does not rule on |
| 2 | R2 with **real-valued arms only** at `n = 32768` | 4.130 GiB, 38% headroom | 3.36 h | the phase arm — the arm the round exists to measure — is absent from its deciding cell |
| 3 | a memory-cheaper complex operator | — | — | **a new construction; the round's first law strikes it** |

Option 1 is **cheaper as well as feasible** — 5.29 h against the 10.38 h the
plan currently books for R2 — which is worth stating plainly, because the
schedule improves under the option that also fits.

`[ASSUMED]` throughout §3.4: that it.5's phase-gate arm is `ceq/arm_phase.py`
as it stands. If it.5 rebuilds it, **every `arm_phase` number in this
certificate must be re-measured with this probe before it.7** — the call
`V15_NEPTUNE_SYSTEMS.md` §8 made for the composed arm, and it was right.

---

### 3.5 The dtype the model is priced at, asserted rather than assumed

**The shipped arm is `complex64`, not `complex128`, and this certificate is
priced at `complex64`.** §3.2's bytes-per-element column is **read off the live
operator tensor** (`a.element_size()`), not inferred from `ceq/arm_phase.py`'s
`CDTYPE = torch.complex128`: `nn.Linear` builds at `float32`, `_ctype(float32)`
returns `complex64`, and the probe measured **`complex64`, 8 B/element**.
`DTYPE`/`CDTYPE` govern `label_cell` and the five binds, not the trained
forward — the same conclusion `V16_R1_DEVICE_READY.md` reached from the source
side, arrived at here from the allocator side. **Had it been `complex128` the
operator term would double again and the phase arm's cap would fall from
`n ≈ 22,100` to roughly `n ≈ 11,600`.**

**TF32 IS ASSERTED OFF, and the assertion is a measurement.** `[MEASURED]`

| flag | value on this box | governs |
|---|---|---|
| `torch.backends.cuda.matmul.allow_tf32` | **`False`** | the operator's matmuls — **this is the one that matters** |
| `torch.backends.cudnn.allow_tf32` | `True` | cudnn convolutions; this arm has no cudnn-backed op, so it is inert here |
| `torch.get_float32_matmul_precision()` | `highest` | the same setting through the newer API, and consistent with the first row |

**TF32 propagates into `complex64` matmul on this card**, and this node
measured it on the **shipped arm** rather than on a bare matmul. Enabling it
moves the operator by **`3.184e-04` relative** and the arm's forward output by
**`3.064e-04`** `[MEASURED]`. `V16_R1_DEVICE_READY.md` measured `2.958e-04` on
the bare product against `3.237e-07` with it off; **two independent
measurements at different levels of the stack agree to within 8%**, which is
what makes this a property of the card rather than of either probe. Three
orders of magnitude, from one global flag, **with no dtype in any record
changing**.

The measurement had to be taken at `identity_heads()`, and the reason is
§5.4's: the as-constructed arm's operator already contains non-finite values,
so a TF32 delta measured there reads `nan` and says nothing about TF32. That
is a third independent sighting of the same defect, arrived at while trying to
measure something else.

**So the certificate carries the flag as a condition, not as an observation.**
`allow_tf32 = False` is torch's default for matmul and nothing in `scale/` or
`ceq/` sets it — but nothing *asserts* it either, and a default is not a
guarantee across a torch upgrade or a stray line in a runner. **§6.3 requires
it journalled beside `device` and `deterministic_algorithms`**, for the same
reason those two are journalled: an unrecorded global that moves the arithmetic
by `1e-4` is a cell-identity field whether or not anyone has written it down.

---

## 4. THE WALL-CLOCK LAW, RE-FITTED

### 4.1 The law

Median of the timed steps (forward + backward + `Adam.step`), 2 warm-up steps
discarded, `s = 64`, random data, `threads = 8`, softmax arm. Median and not
mean: §2 is the reason to care.

| `n` | CPU s/step `[MEASURED]` | ×150 | CUDA s/step `[MEASURED]` | ×150 | CPU/GPU |
|---|---|---|---|---|---|
| 2,048 | 0.0858 | 12.9 s | 0.0110 | 1.6 s | 7.8× |
| 4,096 | 0.1808 | 27.1 s | 0.0201 | 3.0 s | 9.0× |
| 8,192 | 0.4098 | 61.5 s | 0.0403 | 6.0 s | 10.2× |
| 16,384 | 0.9009 | 135.1 s | 0.0804 | 12.1 s | 11.2× |
| **32,768** | **1.8818** | **282.3 s** | **0.1603** | **24.0 s** | **11.7×** |

**Fitted laws, log-log OLS over all five points** `[FITTED]`:

| device | law | exponent | R² |
|---|---|---|---|
| CPU, 8 threads | `s/step = exp(−11.0236) · n^1.1228` | **1.1228** | **0.999704** |
| CUDA | `s/step = exp(−11.9670) · n^0.9734` | **0.9734** | **0.999384** |

**Against what would otherwise have been carried.** `secs ~ n^1.338`
`[INHERITED, V13_DAG_TASKLIST.md:88-91]` was retired last round. The previous
round's own re-measurement — CPU `n^1.1920`, CUDA `n^0.9734` — is what L-TIME
forbids this round from carrying. **The CUDA exponent reproduces to four
decimal places (0.9734 against 0.9734). The CPU exponent does not (1.1228
against 1.1920)** — and §2 is why: a CPU exponent fitted across five sizes on a
box whose per-step cost varies by 3.3× with thread placement is a fit through a
moving target. Across this probe's eight runs the CPU exponent read 1.1034 to
1.1532, a spread of **0.050**; the CUDA exponent read 0.9693 to 0.9740, a
spread of **0.005**. **The CUDA law is ten times the more stable object, which
is §2.6.3's schedulability finding in a second currency — and it is why the
CUDA column of §6.2 is a projection while the CPU column is a range.**

**The CUDA exponent near 1 is the signature of a memory-traffic-bound kernel**
on a materialised `[n,s,s]` at fixed `s`. FLOP-per-token is not offered as a
substitute: `sizing.flops_per_token` models an LM, not this arm.

### 4.2 The arm multipliers

The law is fitted on softmax; every other arm is priced as softmax × a
multiplier measured at one shape on the certified device.

| arm | CUDA s/step at `n = 2048` `[MEASURED]` | multiplier |
|---|---|---|
| softmax | 0.01000 | ×1.000 |
| arm_pl | 0.01043 | **×1.042** |
| **arm_phase** | **0.02088** | **×2.085** |

**The complex arm costs 2.085× a real one in time and 1.94× in bytes** (§3.1's
per-example figures, 254,479 against 131,209). The two ratios being that close
says the arm is bandwidth-bound on the same tensor in both currencies, which is
the consistency check that makes a single multiplier defensible across `n`.

---

## 5. WHAT CUDA CERTIFICATION REQUIRES

### 5.0 A moving target, read rather than asserted

`scale/negation_scope.py` and `scale/r10_capacity_sweep.py` changed **during**
this probe's development: `calibrate_bar` gained a `device=` parameter between
the probe's first and second runs, and `r10_capacity_sweep.py`'s
`--device cuda` abort was deleted between its second and third. Both are now
committed. §7's section E therefore **reads their state at run time and prints
what it found**, which is the only form of that claim that survives a
concurrent edit. Everything in §5 carries that caveat.

### 5.1 What the blocker was

`V15_MERCURY_DEVICE.md` (e): `main()` aborted before opening a journal, exit 1,
because `negation_scope.calibrate_bar` built its reference batch, its
positive-control net and its feature tensor on CPU unconditionally, and
`GATE_TOL = 1e-3` was measured over 16 untrained **CPU** seeds (M-14). Scoring
a CUDA cell against either carries a threshold across a device boundary, which
is `MISTAKES.md` **V-22**.

### 5.2 What it takes, itemised, and what each costs

| # | requirement | state at run time `[MEASURED]` | cost |
|---|---|---|---|
| 1 | `calibrate_bar` accepts and honours `device` | **DONE** — batch built on device; control net constructed on the seeded CPU generator then `.to(device)`, so weights are identical values on both devices | 0 (landed) |
| 2 | the caller threads `device` into it | **DONE** — `passes device= into calibrate_bar : True` | 0 (landed) |
| 3 | the `--device cuda` abort deleted | **DONE** — `still aborts on --device cuda : False` | 0 (landed) |
| 4 | the five bar clauses **re-run on cuda**, `bar_verdict` recorded | **DONE** — `V16_BAR_RECERT.md`, landed while this probe ran | **0.447 s** per call |
| 5 | **`GATE_TOL = 1e-3` re-measured on cuda** over 16 untrained seeds, as M-14 measured it on cpu | **DONE** — same node | **0.32 s** for 16 seeds |
| 6 | no cross-device pooling | **DONE** — `refuse_cross_device_pool` refuses any spread; `device` journalled on every record | 0 (landed) |
| 7 | nondeterminism declared | **DONE but WRONG for this round's arms** — §5.3 | see §5.3 |

**TOTAL COMPUTE TO RE-CERTIFY: 0.77 seconds.** Not hours — **seconds**. **The
13× decision was never gated on compute. It was gated on a code edit and a
measurement nobody had run**, and both landed in this same iteration —
`V16_BAR_RECERT.md` certifies the bar and the 0-step gate on cuda, and every
item in this table is now DONE. **What is left is not a measurement, it is the
regime mismatch §5.3 found between that certification and the arms.**

The corpus makes it cheaper than it looks: every `M3_TASKS` builder draws on a
**seeded CPU generator and only then moves** (`negation_scope.py:96`,
`:426–427`), so the corpus **bytes are bit-identical on both devices by
construction**. Only the arithmetic differs — which is exactly the quantity
re-certification measures.

### 5.3 The conflict: the bar's regime is one the arms cannot execute in

`scale/r10_capacity_sweep.py` sets, unconditionally, on the cuda path:

```
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
```

Measured per arm, in separate subprocesses (the env var must be set before the
CUDA context exists, so an in-process toggle raises instead of measuring —
which is what this probe's first attempt did):

| arm | det OFF | det **STRICT** | det `warn_only=True` | warn/off |
|---|---|---|---|---|
| softmax | 0.01099 | 0.01172 | 0.01298 | 1.182× |
| **arm_pl** | 0.01152 | **BLOCKED** | 0.01360 | 1.181× |
| **arm_phase** | 0.02117 | **BLOCKED** | 0.02786 | 1.316× |

`[MEASURED]`, seconds per step at `n = 2048`, `s = 64`, CUDA.

```
RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation,
but you set 'torch.use_deterministic_algorithms(True)'.
```

**Every gated arm in this round is a prefix scan, and `cumsum` has no
deterministic CUDA kernel.** The blocking sites are `ceq/arm_pl.py:90` and
`ceq/arm_phase.py:121-122, 128, 138`. The scan is not incidental to these arms:
`C_j = Σ_{k≤j} log m_k + i Σ_{k≤j} θ_k` **is** §S-M′.

**How this was missed, and it is a named class.** `V15_MERCURY_DEVICE.md` (b.3)
measured the determinism cost at `0.921×` and concluded "no slowdown, so it is
set unconditionally rather than left off to save a cost that does not exist".
That measurement was taken *on the softmax arm* — **the one arm in the table
above that strict mode does not block**. A constant validated on one shape and
applied to another is the same class as V-22 and as the `C_OPERATOR` transfer
§3.2 corrects. The conclusion was right for the arm measured and wrong for
every arm the round runs.

**And this is not a probe artefact — it is the round's live conflict.**
`V16_BAR_RECERT.md` certified the bar **under `use_deterministic_algorithms(True)`**,
and `V16_R1_DEVICE_READY.md` then measured that `cumsum_cuda_kernel` is *the
only* nondeterministic op either arm touches and that every scan, key-bias,
phase-factor, normalizer, winding and operator path **RAISES** under that flag.
So the setting the instrument was certified in is the setting the arm cannot
execute in. Two committed nodes, each correct, and their conjunction blocks
R1′ on CUDA. This node's job is to price the exits, not to choose by taste.

**`warn_only=True` costs nothing, and the measurement says so by refusing to
settle.** Across three runs the `warn_only`/off ratio read `1.06/1.06/1.26`,
`1.06/0.95/1.09` and `1.18/1.18/1.32` — **including a value below 1**, which a
real cost cannot produce. That is the right answer rather than a noisy one:
`warn_only` changes what happens *when* a nondeterministic kernel is reached,
not *which* kernel is selected, so it cannot move throughput, and the spread is
the box's own (§2.3). `V15_MERCURY_DEVICE.md`'s `0.921×` — also below 1 — is
the same non-effect measured once and read as a result. **Exit 2 is therefore
free, which is the single most important thing about it.**

### 5.3.1 The three exits, priced

**The evidence they are priced against, from three independent measurements:**

| measurement | source | result |
|---|---|---|
| `cumsum` float64 1-D, `n = 64` and `n = 10 000`, 20 repeats | `V16_R1_DEVICE_READY.md` | **bitwise** |
| `cumsum` float64 1-D, `n = 1 000 000`, 20 repeats | `V16_R1_DEVICE_READY.md` | **`6.82e-13`** — the hazard is real |
| the same 1-D case, independently | §7 section I, this node | **`1.82e-12`** — same order, reproduced |
| `cumsum(dim=-1)` at **R1′'s shape `[2048, 64]`**, 20 repeats | §7 section I | **`0.000000e+00`** |
| `cumsum(dim=-1)` at **R2's exact shape `[32768, 64]`**, 20 repeats | §7 section I | **`0.000000e+00`** |
| `cumsum(dim=-1)` at **8× R2, `[262144, 64]`** — 16.8M elements | §7 section I | **`0.000000e+00`** |
| `cumsum(dim=-1)` at **`[512, 4096]`** — R2's element count, 64× the reduction | §7 section I | **`0.000000e+00`** |
| **gradient** of a scan arm, whole forward+backward, 4 repeats at `n = 2048` | §7 section G | softmax `0.000e+00`, **arm_pl `0.000e+00`** |
| **gradient** of a scan arm at **R2's shape `n = 32768`**, 8 repeats | §7 section I | **`0.000000e+00`** |

**The batched shape was the gap in the evidence, and closing it changes the
answer's status.** The sibling node measured a one-dimensional tensor;
`arm_pl.scan` and `arm_phase.scan_phase` call `cumsum(dim=-1)` on `[n, s]` with
a **reduction length of 64** and a **launch of up to 2.1 million elements**.
Whether the hazard follows the reduction length or the element count decides
whether the narrow claim covers R2 at all. **Measured: it follows the reduction
length.** At 16.8 million elements with a reduction of 64 the result is
bitwise; at 1 million elements with a reduction of 1,000,000 it drifts; and at
R2's *element count* with a 4,096-long reduction it is still bitwise. **The
arms reduce over 64, which is two to four orders of magnitude below anything
that moved**, and the boundary is bracketed between reduction length 4,096
(bitwise, 20 repeats) and 1,000,000 (`1.8e-12`).

**That is a mechanism, not a coincidence of scale**, and it is what exit 2's
claim needed: `cumsum`'s CUDA kernel switches to a parallel multi-block scan
only when the reduction is long enough to need one, and a 64-long reduction
never leaves the deterministic single-pass path however many rows are launched
at it. The claim *"reproducible at the shapes these arms use"* now has a stated
reason and a measured boundary rather than an assumption — **and the strongest
form of the evidence is the last row of the table: the whole gradient of a scan
arm at R2's exact `n`, bitwise identical over 8 repeats.** That is the quantity
a verdict is built from, not a bare op.

| # | exit | compute cost | schedule cost | what it costs in evidence |
|---|---|---|---|---|
| **1** | **Run CUDA with the flag off.** | 0 | **68.35 GPU-h** unchanged | **A named V-22-adjacent gap.** The bar's five clauses were certified under `use_deterministic_algorithms(True)`; the cells would run without it. That is a constant certified under conditions the reading does not reproduce — the same shape of defect as V-22, one level up. **If this exit is taken it must be written into `MISTAKES.md` as such, not mentioned in a Limits paragraph.** |
| **2** | **`warn_only=True`, and certify the narrow empirical claim with a boundary.** | §7 section I, seconds | **68.35 GPU-h — the same as exit 1.** `warn_only` cannot change kernel selection, and the measured ratio straddles 1.0 in both directions across three runs | **The honest middle, and it now has a must-fire rather than an assumption.** The claim is *"`cumsum` is bitwise reproducible at the shapes these arms use"*, and its boundary is measured: 1-D `n = 1e6` drifts `6.82e-13`, and §7 section I extends the boundary search to the batched shapes. **The flag stays on, journalled, with `cumsum` exempted by warning — so the bar's certification regime and the run's regime are the same flag at the same setting, which is exactly what exit 1 gives up.** |
| **3** | **CPU for R1′.** | 0 | **5.61 CPU-h** at the 9600 ladder, **0.09 h** at the 150-step floor — R1′ alone is the cheapest node in the plan | The 12.12× goes away *for that node only*. R1′ on CPU and R2 on CUDA is legal — each verdict is against a closed-form floor (§5.5.2) — but nothing may pool or difference across them, and §2.6.2's 2.0× placement band applies to R1′'s wall clock. **Generalised: every node at `n ≤ 2048` on CPU costs 25.87 CPU-h and leaves 65.06 GPU-h**, which is a defensible split because the `n ≤ 2048` nodes are exactly the cheap ones. |

**RECOMMENDATION: exit 2.** It is the only one of the three that leaves the
bar's certification regime and the run's regime identical, it converts an
assumption into a claim with a measured boundary, and **its schedule cost is zero**, which the
first draft of this table got wrong by treating a straddling ratio as a price. **Exit 1 is cheaper by exactly the
amount of evidence it discards, and that is the R11 trade this round is meant
to stop making.**

**The conflict does not by itself decide the device.** Exits 1 and 2 both keep
CUDA; only exit 3 moves a node to CPU, and it moves the cheapest node. **The
13.6× the contract names is not at risk from the determinism conflict** — it is
at risk from §3.4, which is a memory fact and has nothing to do with
determinism.

### 5.4 The precondition every hour in §6 depends on

While pricing the phase arm this node measured whether it can take a gradient
step. It cannot, and the failure is device-independent.

| arm | init | cpu | cuda |
|---|---|---|---|
| softmax | as constructed | none in 40 steps | none in 40 steps |
| arm_pl | as constructed | none in 40 steps | none in 40 steps |
| arm_pl | `zero_heads()` | none in 40 steps | none in 40 steps |
| **arm_phase** | **as constructed** | **step 0** (50.0763% of positions at `m=0`) | **step 0** (49.8627%) |
| **arm_phase** | **`identity_heads()`** | **step 14** (0.0031%) | **step 23** (0.0031%) |

*First step at which the gradient over all 4,820 parameters is non-finite,
40-step budget, `n = 512`, random data.* `[MEASURED]`

**At construction, half of all positions sit at `m = 0` and the gradient is
non-finite before the first optimizer step.** `ceq/arm_phase.py:121,128` take
`log(m)` with `m = clamp(u, 0, 1)`, and `u` is a randomly-initialised linear
head, so roughly half its outputs are negative and clamp to the **closed** lower
endpoint. `log(0) = −inf`; its backward is `1/0`.

`identity_heads()` starts at `m = 1` and survives 14 steps (cpu) or 23 (cuda)
before a **0.0031%** fraction of positions reaches the endpoint and the same
`−inf` appears. That the step differs by device and the outcome does not is the
signature of a numerical boundary event rather than a device defect.

**The closed endpoint is not a bug in the arm; it is the X₃₆ delta's stated
design.** `magnitude`'s own docstring: *"`clamp` and not `sigmoid`: the
endpoints must be ATTAINED … so `|a| ≤ 1` is a property of the arithmetic and
not of the parameter values a run happens to reach."* The arm reaches the value
it was built to be able to represent, and the gradient does not survive it.

**This node does not fix it.** `ceq/` is outside this write scope, and a
reformulation would be a new construction the round's first law strikes. It is
filed as a **precondition on it.5**, ahead of the arm being trained at it.7:
every hour in §6.2 assumes an arm that completes its steps. The natural repair
is a floor on the magnitude *inside the `log` only*, which changes no attainable
**value** of `m` and is a numerical guard rather than an architectural change —
but that judgement belongs to whoever owns `ceq/arm_phase.py`.

### 5.5 The conditions that carry forward

`V15_NEPTUNE_SYSTEMS.md` §6.3's three conditions, restated where this round
changes them:

1. **No CUDA cell may be pooled with a CPU-taken cell.** M-10 measured thread
   count alone moving `eval_nrmse` by `2.345e-3`, which is `0.464` of
   `Δ_eq = 0.005051`; device is a larger perturbation.
   `[INHERITED, MISTAKES.md M-10]` **What is new:** 142 of 158 journalled
   `cell` rows carry no `device` field — only R1's 16 do.
   `refuse_cross_device_pool` reads a missing field as `"cpu"`, which is
   correct by provenance (the flag did not exist), and refuses any spread. The
   guard is sound; the point is that those 142 rows' device lives in that
   default rather than in the record.
2. **R1′ on CUDA is a legitimate floor-crossing verdict and an illegitimate
   cell-by-cell comparison to R1.** `floor₁ = 0.7071` is a closed form in `t*`
   with no device in it, so an 8-seed CUDA reading may be read against it. What
   may not happen is pooling R1's eight CPU seeds with R1′'s eight CUDA seeds
   into sixteen, or reading a per-seed delta between them. **And the softmax
   control cannot be inherited from R1 — it must be re-taken on the certified
   device**, which is why §6.1 gives R1′ two arms and not one.
3. **`scale/it11_verdict.py::by_seed` buckets by `threads` alone**, and on CUDA
   the thread count does not name the reduction lane. It degrades safely — it
   picks the larger bucket and reports what it dropped, or raises on a
   same-seed disagreement — but the protection against device mixing is
   `refuse_cross_device_pool`, which a consumer must actually call.
   `[INHERITED, V15_MERCURY_DEVICE.md Limits]`
4. **Two published numbers already differ across devices, and neither is a
   bind.** `V16_R1_DEVICE_READY.md` measures a summation-order census at
   `9767` exactly-one prefixes on cpu against `7713` on cuda — with zero above
   one on both, so the bound the census exists to check holds on both — and a
   complex-gemm gap of `1.11e-16` on cpu against **exactly `0.0`** on cuda,
   cuBLAS being the stricter of the two. Both are pinned per device.
   **These are the calibration for every later device difference:** a
   difference that changes a count while leaving the property intact, and a
   difference that favours the new device, are what harmless device-dependence
   looks like. A difference that does neither is the one to stop on.
   `[INHERITED, V16_R1_DEVICE_READY.md §4.1]`

---

## 6. THE CERTIFICATE

### 6.1 The cell census, and every assumption in it

A **cell** is one `(arm, bed, t*, n, seed)` training run. `N = 8` is
contract-fixed (PART IV). **Arms-per-node is not**, so each row names the
assumption that produced its arm count. `P` = phase-gate arm (complex
operator); `S` = a real-valued softmax-class arm.

| it | node | arms | cells | `n` | the assumption |
|---|---|---|---|---|---|
| 7 | **R1′** BED-M `t*=2` | `P`,`S` | 16 | 2,048 | contract-fixed `n`. Two arms because the softmax control **cannot be inherited from R1** — R1 is CPU-taken and §5.5.2 forbids the cross-device comparison |
| 10 | **R2** BED-M `t*=8` | `P`,`S` | 16 | 32,768 | contract-fixed `n`; **`P` does not fit — §3.4** |
| 10 | **R-SKY** @ R1′ | `S` | 8 | 2,048 | the skyline is a *trained* arm: it.9 specifies `m=93, 4,769` parameters, and 4,769 is exactly softmax's count `[MEASURED]`, so it is param-matched and priced as softmax-class |
| 10 | **R-SKY** @ R2 | `S` | 8 | 32,768 | *"read beside R1′/R2"*, so both scales |
| 12 | MARS param-match control | `S` | 8 | 2,048 | attack #6 files a *"param-matched even-feature-only control column"*; the census and leak attacks are non-training |
| 17 | **R3(a)** delay bed | `S`,`S`,`S`,`P` | 32 | 32,768 | four arms named by PART IV; the composed arm priced as complex — **NOT MEASURED**, pessimistic end |
| 18 | **R3(b)** power-law bed | `S`,`S`,`S`,`P` | 32 | 32,768 | same four arms |
| 20 | **R4** skylines, both beds | `S`,`S` | 16 | 32,768 | R4 is a table over existing cells, but *"each arm vs its bed's native skyline"* needs a BED-K skyline per bed, and neither exists yet |
| 25 | **R5** BED-1 | `P`,`S` | 16 | 32,768 | S5′ exit accuracy plus its control; CK, Morse, Pesin and `β₁` are analyses of the trained artifacts |
| 28 | **R6** hidden-cause | `S`×4 | 32 | 2,048 | three noise levels plus the no-plant flatness control, which PART IV puts **first**; priced as trained, the pessimistic reading of an estimator |
| 29 | **R7** `ζ>0` vs `ζ=0` | `P`,`S` × 3 rungs | 48 | 512 / 1,024 / 2,048 | *"`≥2×` fewer examples to R1′'s crossing"* is an `n`-sweep, and R1′'s `n` is 2,048, so the rungs sit at and below it |
| | | | **232** | | |

`[ASSUMED]` on `n` wherever the contract is silent: R3, R5 and R6 inherit R2's
`n` so that R4's two-sided table compares two beds at one resolution —
`V15_NEPTUNE_SYSTEMS.md` §2.5's argument, unchanged and still the cheapest way
to make R4's sentence mean anything.

**One node is deliberately NOT in the table, and D-CALIB says it should be
planned for.** it.19: *"if the composed arm loses (a): the summary bandwidth
sweep, stated as a dial."* `V16_CALIBRATION.md`'s **D-CALIB-1** makes the
author's counter-prediction the expected outcome, and R3's counter is exactly
*"composed loses on (a), matches on (b)"*. So the sweep is the **base case**,
not the tail. Priced separately at `[ASSUMED]` 4 dial settings × 8 seeds with
the composed arm: **28.08 GPU-h** at `n = 32768` and the full ladder, or
**14.30 GPU-h** at the `n = 16384` that §3.4 forces on a complex arm
`[FITTED]`. It is excluded from §6.2's total only because the contract does not
list it as a cell, and it is named here so the exclusion is a choice rather
than an oversight. **At 28.08 h it would be 41% of the whole plan.**

### 6.2 The projection

`[FITTED]` from §4's laws and §4.2's per-arm multipliers.

| budget | cells | **CUDA h** | **CPU h** | CPU/CUDA |
|---|---|---|---|---|
| **150-step floor** | 232 | **1.068** | **12.75** | 11.94× |
| **full 9600 ladder** | 232 | **68.35** | **815.88** | 11.94× |
| *pessimistic* (CPU ×2.55, CUDA ×1.05 — §2.3's measured per-device spreads) | | **71.53** | **2,077.46** | **29.0×** |

The four largest lines at the 9600 ladder, CUDA:

| it | node | cells | CUDA h | CPU h |
|---|---|---|---|---|
| 17 | R3(a) delay bed | 32 | 17.11 | 207.71 |
| 18 | R3(b) power-law bed | 32 | 17.11 | 207.71 |
| 10 | R2 | 16 | 10.38 | 126.06 |
| 25 | R5 | 16 | 10.38 | 126.06 |

**The point estimate and its pessimistic end, reported together as the R11
bias requires.** The contract's D-DEVICE calls this "a 13× decision"; measured,
it is **11.94×** at the point estimate and **29.0×** at the pessimistic end —
and the pessimistic end is *worse for CPU*, because the contention and
placement spread this node measured falls almost entirely on the host lane.
**The contract's own figure is the optimistic reading of the decision.**

**No numeric shrink factor is applied to any of this**, per `V16_CALIBRATION.md`
**D-CALIB-3** (*"sign, never size"*). The pessimistic column is a measured
spread, not a discount.

**68.35 GPU-hours is under three days of AFK loop.** 816 CPU-hours is 34.0
days and 2,077 is 86.6 days; neither is a schedule. **At the 150-step floor the
CPU plan is 12.75 h and is affordable** — which matters, because R1 ran at 150
steps, so the floor is the budget the round has actually been buying.

**Two adjustments the tables do not contain, both stated:**

- **§3.4's substitution improves the total.** Moving the four `arm_phase` slots
  from `n = 32768` to `n = 16384` — which §3.4 shows is not optional — saves
  **13.78 GPU-h** and brings the 9600-ladder total to **54.58 h** `[FITTED]`.
- **it.19's bandwidth sweep, if D-CALIB-1 is honoured, adds 14.30–28.08 GPU-h.**
  Netting the two: **68.9–82.7 GPU-h**, i.e. the plan lands roughly where the
  naive total already put it, for entirely different reasons.

### 6.3 THE CERTIFICATE, SIGNED

> **DEVICE: CUDA** — NVIDIA GeForce RTX 4060 Laptop GPU, sm_89, 24 MPs,
> 7.996 GiB total / 6.939 GiB free, `torch 2.5.1+cu121`,
> `torch.use_deterministic_algorithms(True, warn_only=True)` with
> `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `device` journalled on every record,
> `refuse_cross_device_pool` called before any aggregate.
>
> **MEMORY MODEL.** `ceq/sizing.py`'s constants, validated:
> `measured/predicted = 0.950` for real-valued arms across `n = 512…32768`.
> **For a complex arm the module is not valid** — use
> `C_OPERATOR = 7.50` at `ob = 8.0`, or the measured **258,609 B/example**
> at `s = 64`, which is what §3.4's caps rest on.
>
> **WALL-CLOCK LAW.** `s/step = exp(−11.9670) · n^0.9734`, **R² 0.999384**,
> five points over `n = 2048…32768`, softmax arm, `s = 64`.
> Arm multipliers ×1.000 / ×1.042 / **×2.085** for softmax / ARM PL /
> ARM PHASE.
>
> **40-ITERATION PROJECTION.** 232 cells: **68.35 GPU-h** at the 9600 ladder,
> **1.07 GPU-h** at the 150-step floor. **Pessimistic end 71.53 GPU-h.**
> Against 815.88 / 12.75 CPU-h, pessimistic 2,077.46 — **an 11.94× decision at
> the point estimate and 29.0× at the pessimistic end.**
>
> **CELLS THAT DO NOT FIT:** it.10 R2, it.17 R3(a), it.18 R3(b), it.25 R5 —
> each at `n = 32768` **with the phase-gate arm** (and, for R3, with the
> composed arm if it is complex, which is NOT MEASURED). Cap `n = 22,106` at a
> 20% margin, `n = 27,633` at zero, last power of two **`n = 16384`**.
>
> **ARITHMETIC ASSERTED, not observed:** `allow_tf32 = False` for matmul,
> operator dtype `complex64` at 8 B/element, `deterministic_algorithms` as
> §5.3.1 exit 2 sets it. **All three journalled beside `device` on every
> record**, because each of them moves the arithmetic and none of them is in
> a cell's identity today.
>
> **SIGNED CONDITIONAL ON FOUR THINGS, each of which is someone's named task:**
> 1. **The determinism conflict resolved by §5.3.1 exit 2** —
>    `warn_only=True` at **zero measured cost**, with the narrow claim carrying
>    §7 section I's boundary: bitwise over 20 repeats at reduction length 64
>    and 4,096 and at up to 16.8M elements, drifting `2.27e-12` only at a
>    reduction of 1,000,000. Under exit 1 the certificate still stands but the
>    round carries a **V-22-adjacent gap that must be written into
>    `MISTAKES.md`**, not into a Limits paragraph. Under exit 3 R1′ moves to
>    CPU at 5.61 CPU-h and the rest of the plan is unaffected.
> 2. `V16_BAR_RECERT.md`'s bar and `GATE_TOL` **read in the same regime the
>    cells run in** (§5.2 items 4–5, 0.77 s of compute). The bar is certified;
>    what is conditional is that its regime and the run's regime match, which
>    is exactly what condition 1 decides. **If the bar reads BROKEN on cuda in
>    the run's regime, this certificate reverses to CPU** and §6.2's CPU column
>    with its 2.55× spread is the schedule.
> 3. **R2's `n` reconciled with §3.4** — `n = 16384` for the phase arm, or the
>    phase arm dropped from R2. Not a device condition; it binds on both
>    devices and it binds before it.10.
> 4. The phase arm's `log(clamp(u,0,1))` made finite (§5.4), or **no hour above
>    is spendable on it at all.**
>
> **THE CPU FALLBACK, if condition 2 fails:** the 150-step floor at 12.75 h is
> affordable and the 9600 ladder at 816 h is not. On CPU the round buys the
> floor for every node and the ladder for none, **with CPU affinity pinned and
> journalled** (§2.6) — without which the schedule has a free parameter in it
> and no cell is comparable to any other. **That band is 3.3× inside this
> probe's own run of record and 5.8× if R1's cell counts** (§2.6.2).

---

## 7. THE PROBE OUTPUT, VERBATIM

`python scripts/v16_device_probe.py`, repo HEAD `e292469`.

```
==============================================================================
A. ENVIRONMENT
==============================================================================
torch 2.5.1+cu121   cuda_available=True
gpu            NVIDIA GeForce RTX 4060 Laptop GPU  sm_89  MPs=24
vram total     7.996 GiB (8585216000 B)
vram free      6.939 GiB (7451181056 B)
host RAM       15.71 GiB total, 9.24 GiB available (41.2% used)
cpu            28 logical; torch reads 2 threads after imports (scale/m3_capability.py:64 pins 2 at import; every timing below sets its own pin)
box at rest    cpu 6.4% busy, 293 processes
cell shape     s=64 d_model=16 layers=1 heads=1, FULL BATCH (batch==n)

==============================================================================
B1. R1's GAP -- THREAD COUNT (n=2048, s=64, softmax, randn, cpu)
==============================================================================
V15_R1.md ran threads=8; V15_NEPTUNE_SYSTEMS.md section 4.2 ran 12.
 threads  median s/step        R1 stat         x150 s
       2         0.1151         0.1166           17.5
       6         0.0837         0.0893           13.4
       8         0.0861         0.0871           13.1
      12         0.0814         0.0822           12.3
      20         0.0775         0.0791           11.9
threads=8 / threads=12 = 1.058   (a 5.8x gap needs 5.8)

==============================================================================
B2. R1's GAP -- DATA SOURCE and ARM, at R1's own pin and step count
==============================================================================
150 steps, threads=8, cpu. 'R1 stat' is total/steps, the statistic
scripts/v15_r1.py:train_one records as `secs`, warm-up included.
arm        data         median     R1 stat      x150 s   V15_R1.md filed
softmax    randn        0.0810      0.0822        12.3   -
softmax    corpus       0.0819      0.0829        12.4   71.32 s -> reproduced at 0.17x
arm_pl     randn        0.0876      0.0880        13.2   -
arm_pl     corpus       0.0900      0.0897        13.4   91.63 s -> reproduced at 0.15x
arm_phase  randn        0.1831      0.1859        27.9   -
arm_phase  corpus       0.1825      0.1835        27.5   -

==============================================================================
B3. R1's GAP -- HOST CONTENTION (n=2048, softmax, randn, threads=8)
==============================================================================
k = concurrent 1-thread torch processes on 2048x2048 matmuls, each
holding `hold` GiB of touched ballast. Two axes: cores+bandwidth,
and host memory. R1's gap needs 5.8x from somewhere.
The CUDA column is the SAME cell on the GPU under the SAME host
pressure: the schedulability half of the device decision.
   k   hold  cpu s/step   cpu x150    vs k=0 cuda s/step    vs k=0 RAM free
   0   0.00      0.0798       12.0     1.00x      0.0110     1.00x    8.10G
   4   0.00      0.1300       19.5     1.63x      0.0114     1.03x    8.08G
   8   0.00      0.1928       28.9     2.42x      0.0114     1.04x    8.13G
  12   0.00      0.1179       17.7     1.48x      0.0115     1.05x    8.11G
   4   0.75      0.1319       19.8     1.65x      0.0112     1.02x    8.09G
   8   0.75      0.2031       30.5     2.55x      0.0114     1.04x    8.55G
cpu spread over the sweep 2.55x;  cuda spread 1.05x

==============================================================================
B5. R1's GAP -- WHICH CORES THE 8 THREADS LANDED ON
==============================================================================
This box is an Intel i7-14700HX: 20 physical cores, 28 logical, which
is 8 P-cores (SMT, logical 0-15) plus 12 E-cores (logical 16-27).
`torch.set_num_threads(8)` names a COUNT, never a placement, so the
Windows scheduler decides -- and it demotes work it reads as
background. A cell whose 8 threads land on E-cores is a different
machine from one whose 8 land on P-cores, at identical settings.
Crossed with B3's contention axis, because neither factor alone
reaches R1's number and the two do not compose multiplicatively.
lane                        k median s/step     x150 s  vs P-core
P-cores, SMT siblings       0        0.0912       13.7      1.00x
P-cores, 1 thread/core      0        0.0931       14.0      1.02x
default (no pin)            0        0.0818       12.3      0.90x
E-cores only                0        0.1315       19.7      1.44x
default (no pin)           12        0.1142       17.1      1.25x
E-cores only               12        0.2701       40.5      2.96x

worst lane / P-core lane: 2.96x.  V15_R1.md filed 71.32 s against
this P-core lane's 13.7 s, a factor of 5.21. Neither placement nor
contention reaches that alone; the CROSS does most of it, and the
cross is super-multiplicative -- see V16_DEVICE_CERT.md section 2.

==============================================================================
B4. THE SAME SHAPE ON BOTH DEVICES (n=2048, s=64, 150 steps)
==============================================================================
CPU column is B2's randn row, not a second run of the same thing.
arm            cpu med    cpu x150    cuda med   cuda x150   cpu/gpu
softmax         0.0810        12.2      0.0100         1.5      8.1x
arm_pl          0.0876        13.1      0.0104         1.6      8.4x
arm_phase       0.1831        27.5      0.0209         3.1      8.8x

==============================================================================
C1. MEMORY MODEL vs ALLOCATOR -- n sweep at s=64, CUDA, ALL THREE ARMS
==============================================================================
`pred` is `sizing.activation_bytes(arm='signed', dtype='fp32')`, the
module's own number, which knows nothing about a complex operator.
`B/example` is the model-free quantity: both terms are LINEAR in n at
fixed s, so B/example measured at one n IS B/example at every n, and
that -- not the model -- is what the R2 extrapolation rests on.
arm               n     pred GiB     meas GiB      m/p meas B/example
softmax         512        0.066        0.063    0.950         131225
softmax        2048        0.263        0.250    0.950         131209
softmax        8192        1.054        1.000    0.949         131113
arm_pl          512        0.066        0.063    0.950         131233
arm_pl         2048        0.263        0.250    0.950         131211
arm_pl         8192        1.054        1.000    0.949         131114
arm_phase       512        0.066        0.121    1.842         254513
arm_phase      2048        0.263        0.485    1.842         254479
arm_phase      8192        1.054        1.941    1.842         254471

  softmax    B/example over n=512..8192: 131113 .. 131225  (spread 0.09%)
  arm_pl     B/example over n=512..8192: 131114 .. 131233  (spread 0.09%)
  arm_phase  B/example over n=512..8192: 254471 .. 254513  (spread 0.02%)

==============================================================================
C2. TERM SEPARATION PER ARM -- s sweep at n=256, CUDA
==============================================================================
C_OPERATOR is back-solved per arm from s=64 and s=512. The operator's
bytes/element is READ OFF THE LIVE TENSOR, not assumed: ArmPhase is
complex64 at fp32 parameters, which is 2x fp32.
arm        op dtype    B/elt   C_RESIDUAL   C_OPERATOR    meas s=64   meas s=512
softmax    float32         4        17.92        3.499       0.0313       1.0156
arm_pl     float32         4        17.92        3.500       0.0313       1.0159
arm_phase  complex64       8         2.02        7.500       0.0607       3.7669
module values for comparison:  C_RESIDUAL 18.00   C_OPERATOR 3.90

==============================================================================
C3. PREDICT-THEN-PROBE AT R2's SIZE (n=32768, s=64) -- PER ARM
==============================================================================
Predicted with each arm's OWN back-solved constants from C2, against
free VRAM now. Nothing is attempted that the prediction refuses.
free VRAM 6.855 GiB (total 7.996);  x_train adds 4096 B/example
arm            pred act    pred wset     measured        m/p      verdict
softmax           4.005        4.130        4.000      0.999         FITS
arm_pl            4.006        4.131        4.000      0.999         FITS
arm_phase         7.768        7.893            -          -      REFUSED

==============================================================================
C4. LARGEST AFFORDABLE n PER ARM ON THIS GPU
==============================================================================
From C1's measured B/example (the pessimistic end of its own spread)
plus x_train, against free VRAM less a 0.200 GiB eval-set reserve
(x_eval at n_eval=4096 and its no-grad forward).
arm             B/example       n @ 0%      n @ 10%      n @ 20%     last 2^k
softmax            135321        52809        47528        42247        32768
arm_pl             135329        52806        47525        42245        32768
arm_phase          258609        27633        24870        22106        16384

==============================================================================
D1. WALL CLOCK -- CPU, softmax, threads=8, median of timed steps
==============================================================================
       n         s/step    s/150 steps
    2048         0.0858           12.9
    4096         0.1808           27.1
    8192         0.4098           61.5
   16384         0.9009          135.1
   32768         1.8818          282.3
log-log OLS over 5 points: slope 1.1228  R^2 0.999704
log residuals: +0.0065, -0.0259, +0.0141, +0.0235, -0.0182

==============================================================================
D2. WALL CLOCK -- CUDA, softmax, threads=8, median of timed steps
==============================================================================
       n         s/step    s/150 steps
    2048         0.0110            1.6
    4096         0.0201            3.0
    8192         0.0403            6.0
   16384         0.0804           12.1
   32768         0.1603           24.0
log-log OLS over 5 points: slope 0.9734  R^2 0.999384
log residuals: +0.0335, -0.0349, -0.0150, +0.0007, +0.0157

==============================================================================
D3. ARM MULTIPLIER on the certified device (n=2048, CUDA)
==============================================================================
The wall-clock law above is fitted on softmax. Every other arm is
priced as softmax x this multiplier, measured at one shape.
  softmax    0.01000 s/step   x0.999
  arm_pl     0.01043 s/step   x1.042
  arm_phase  0.02088 s/step   x2.085

==============================================================================
E. WHAT CUDA CERTIFICATION ACTUALLY REQUIRES
==============================================================================
negation_scope.calibrate_bar signature, AS IT STANDS IN THE WORKING
TREE (see the provenance note below -- this file is `M` in `git
status` and is owned by another node this iteration):
   (n: 'int' = 2048, s: 'int' = 512, d: 'int' = 256, oracle_fn=None, batch_fn=None, feature_fn=None, steps: 'int' = 150, lr: 'float' = 0.02, seed: 'int' = 0, standardise: 'bool' = True, device=None)
   accepts a `device` argument: True
   the constructions that decide the device, in its body:
      x, y, f, p = (batch_fn or make_batch)(
      g = torch.Generator().manual_seed(seed)
      net = torch.nn.Sequential(torch.nn.Linear(feats.shape[-1], 32),
      net = net.to(device)
   caller state, read off scale/r10_capacity_sweep.py just now:
      passes device= into calibrate_bar : True
      still aborts on --device cuda     : False
      journals `device` on every record : True
      refuse_cross_device_pool present  : True

   cost of ONE calibrate_bar call at the SHIPPED settings
   (n=N_EVAL=4096, s=64, d=24, steps=600, M3_TASKS['e3_t2'] hooks):
      device=None     0.540 s   RAN
      device=cuda     0.447 s   RAN
      Clause VALUES are deliberately NOT printed: reading them is the
      re-certification D-DEVICE assigns to another node, and this
      node prices work rather than performing it.

   0-step RED gate cost, 16 untrained seeds, both devices:
      cpu   2.13 s for 16 seeds
      cuda  0.32 s for 16 seeds

   TOTAL COMPUTE TO RE-CERTIFY ON DEVICE: 0.77 s
   (one calibrate_bar on cuda + 16 untrained cuda seeds for GATE_TOL)

==============================================================================
G. THE PRICE OF DETERMINISM ON DEVICE (n=2048, s=64, CUDA)
==============================================================================
Run in SUBPROCESSES: `CUBLAS_WORKSPACE_CONFIG` must be set before
the CUDA context exists, so an in-process toggle raises instead of
measuring -- which is what the first version of this section did.
Per arm, not per batch, because one arm blocking must not hide the
others -- and one of them does.

arm        det OFF s/step           det STRICT               det WARN_ONLY s/step
softmax    0.01099                  0.01172                  0.01298
arm_pl     0.01152                  BLOCKED: RuntimeError:   0.01360
arm_phase  0.02117                  BLOCKED: RuntimeError:   0.02786

Ratio warn_only/off, the price of the DECLARATION condition 3 asks
for: softmax 1.182x, arm_pl 1.181x, arm_phase 1.316x
V15_MERCURY_DEVICE.md (b.3) filed 0.921x, measured on the SOFTMAX
arm only -- the one arm in this table that strict mode does not
block. L-TIME requires the citing iteration to re-measure; done.

RUN-TO-RUN GRADIENT SPREAD on CUDA at a fixed seed, 4 repeats,
max |grad - grad_0| over all 4820 parameters. This is the quantity
condition 3 protects: kernel noise must not enter the 8-seed sd.
M-10's thread-count floor for comparison: 2.345e-03 on eval_nrmse.
   softmax    warn_only: 0.000e+00
   arm_pl     warn_only: 0.000e+00
   arm_phase  warn_only: nan

==============================================================================
H. CAN THE PHASE ARM TAKE A GRADIENT STEP? (random data, finiteness
   only -- no NRMSE, no seed sweep, no verdict)
==============================================================================
   softmax    as-constructed     cpu   first non-finite gradient: none in 40 steps
   softmax    as-constructed     cuda  first non-finite gradient: none in 40 steps
   arm_pl     as-constructed     cpu   first non-finite gradient: none in 40 steps
   arm_pl     as-constructed     cuda  first non-finite gradient: none in 40 steps
   arm_pl     identity_heads()   cpu   first non-finite gradient: none in 40 steps
   arm_pl     identity_heads()   cuda  first non-finite gradient: none in 40 steps
   arm_phase  as-constructed     cpu   first non-finite gradient: step 0 of 40   (50.0763% of positions at m=0)
   arm_phase  as-constructed     cuda  first non-finite gradient: step 0 of 40   (49.8627% of positions at m=0)
   arm_phase  identity_heads()   cpu   first non-finite gradient: step 14 of 40   (0.0031% of positions at m=0)
   arm_phase  identity_heads()   cuda  first non-finite gradient: step 23 of 40   (0.0031% of positions at m=0)

`m = clamp(u, 0, 1)` and `log(m)` (ceq/arm_phase.py:121,128): the
CLOSED lower endpoint the X36 delta chose on purpose is a `-inf` in
the forward and a `1/0` in the backward. Device-independent -- the
STEP it fires at differs by device, the fact that it fires does not.

==============================================================================
I. THE DETERMINISM BOUNDARY AT THE ARMS' SHAPES, AND TF32
==============================================================================
TF32 state on this box, read live:
   torch.backends.cuda.matmul.allow_tf32      False
   torch.backends.cudnn.allow_tf32            True
   torch.get_float32_matmul_precision()       highest

TF32's effect on the SHIPPED arm (not a bare matmul), at
`identity_heads()`. The identity setting is used BECAUSE the
as-constructed arm's operator already contains non-finite values --
section H's `log(clamp(u,0,1))` at m=0 -- so a TF32 delta measured
there reads `nan` and says nothing about TF32. At identity, m=1 and
log(1)=0, the operator is finite, and the float32 q@k inside the
logits is still exactly where TF32 would bite.
   allow_tf32=False  reference; operator dtype torch.complex64, finite=True
   allow_tf32=True   operator         max relative change 3.184080e-04
   allow_tf32=True   forward output   max relative change 3.064442e-04

cumsum(dim=-1) run-to-run spread on CUDA, 20 repeats, same input.
`reduce` is the reduction length; `batch` is the number of rows.
The arms run reduce=64 at batch up to 32768. R2's exact shape is
marked. A NONZERO spread is the must-fire: the length at which the
narrow claim stops holding.
   reduce     batch      elements   max|rep-first|  note
       64         1            64     0.000000e+00  1-D, the sibling node's short case
    10000         1         10000     0.000000e+00  1-D, sibling's middle case
  1000000         1       1000000     2.273737e-12  1-D, sibling's drifting case
       64      2048        131072     0.000000e+00  R1' shape
       64     32768       2097152     0.000000e+00  **R2's exact shape**
       64    262144      16777216     0.000000e+00  8x R2, to push the launch further
     4096       512       2097152     0.000000e+00  same elements as R2, longer reduction

The same question one level up: the GRADIENT of a scan arm at R2's
shape, 8 repeats, fresh model each time, under warn_only. This is
the quantity a verdict is actually built from, not a bare op.
(arm_pl only -- arm_phase does not fit at n=32768, section C3.)
   arm_pl gradient at n=2048   max|rep-first| over 8 repeats = 0.000000e+00
   arm_pl gradient at n=32768  max|rep-first| over 8 repeats = 0.000000e+00

it11_verdict.by_seed's same-seed disagreement rule is 1e-9; M-10's
cross-thread floor on eval_nrmse is 2.345e-03. Both are printed
beside these numbers in V16_DEVICE_CERT.md section 5.3 so the
comparison is against a stated rule and not against zero.

==============================================================================
F. THE 40-ITERATION PLAN, PRICED PER ARM ON BOTH DEVICES
==============================================================================
Fitted laws: cpu  s/step = exp(-11.0236) * n^1.1228  (R^2 0.999704)
             cuda s/step = exp(-11.9670) * n^0.9734  (R^2 0.999384)
Per-arm multipliers on the softmax law: softmax x0.999, arm_pl x1.042, arm_phase x2.085
`fit?` is C4's cap at a 20% margin: n_max softmax 42247, arm_pl 42245, arm_phase 22106

--- 150-step budget ---
it    node                        cells        n      cpu h     cuda h  fit?
7     R1' BED-M t*=2                 16     2048       0.09      0.011  OK
10    R2 BED-M t*=8                  16    32768       1.97      0.162  NO FIT: arm_phase
10    R-SKY @ R1'                     8     2048       0.03      0.004  OK
10    R-SKY @ R2                      8    32768       0.64      0.053  OK
12    MARS param-match control        8     2048       0.03      0.004  OK
17    R3(a) delay bed                32    32768       3.25      0.267  NO FIT: arm_phase
18    R3(b) power-law bed            32    32768       3.25      0.267  NO FIT: arm_phase
20    R4 skylines, both beds         16    32768       1.28      0.105  OK
25    R5 BED-1                       16    32768       1.97      0.162  NO FIT: arm_phase
28    R6 hidden-cause                32     2048       0.11      0.014  OK
29    R7 zeta rung n=512             16      512       0.02      0.003  OK
29    R7 zeta rung n=1024            16     1024       0.04      0.006  OK
29    R7 zeta rung n=2048            16     2048       0.09      0.011  OK
      TOTAL                         232               12.75      1.068
      pessimistic (cpu x2.55, cuda x1.05)                      32.46      1.118

--- 9600-step budget ---
it    node                        cells        n      cpu h     cuda h  fit?
7     R1' BED-M t*=2                 16     2048       5.61      0.699  OK
10    R2 BED-M t*=8                  16    32768     126.06     10.382  NO FIT: arm_phase
10    R-SKY @ R1'                     8     2048       1.82      0.226  OK
10    R-SKY @ R2                      8    32768      40.82      3.362  OK
12    MARS param-match control        8     2048       1.82      0.226  OK
17    R3(a) delay bed                32    32768     207.71     17.106  NO FIT: arm_phase
18    R3(b) power-law bed            32    32768     207.71     17.106  NO FIT: arm_phase
20    R4 skylines, both beds         16    32768      81.65      6.724  OK
25    R5 BED-1                       16    32768     126.06     10.382  NO FIT: arm_phase
28    R6 hidden-cause                32     2048       7.26      0.905  OK
29    R7 zeta rung n=512             16      512       1.18      0.181  OK
29    R7 zeta rung n=1024            16     1024       2.57      0.356  OK
29    R7 zeta rung n=2048            16     2048       5.61      0.699  OK
      TOTAL                         232              815.88     68.354
      pessimistic (cpu x2.55, cuda x1.05)                    2077.46     71.533

CELLS THAT DO NOT FIT, at any step budget (memory is not a
function of the step count):
  it.10   R2 BED-M t*=8              n=32768 with arm_phase -- cap is n=22106 at 20%, 27633 at 0%; last power of two 16384
  it.17   R3(a) delay bed            n=32768 with arm_phase -- cap is n=22106 at 20%, 27633 at 0%; last power of two 16384
  it.18   R3(b) power-law bed        n=32768 with arm_phase -- cap is n=22106 at 20%, 27633 at 0%; last power of two 16384
  it.25   R5 BED-1                   n=32768 with arm_phase -- cap is n=22106 at 20%, 27633 at 0%; last power of two 16384

==============================================================================
MACHINE-READABLE SUMMARY
==============================================================================
{
 "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
 "vram_free": 7451181056,
 "ram_avail": 9925754880,
 "cpu_law": {
  "slope": 1.1227998768908996,
  "r2": 0.9997040014006797,
  "intercept": -11.023567676348247
 },
 "cuda_law": {
  "slope": 0.9734063827079997,
  "r2": 0.9993836917435757,
  "intercept": -11.966997693137031
 },
 "constants": {
  "softmax": {
   "ob": 4.0,
   "dtype": "float32",
   "C_RESIDUAL": 17.923209054129465,
   "C_OPERATOR": 3.498567853655134
  },
  "arm_pl": {
   "ob": 4.0,
   "dtype": "float32",
   "C_RESIDUAL": 17.923209054129465,
   "C_OPERATOR": 3.499544416155134
  },
  "arm_phase": {
   "ob": 8.0,
   "dtype": "complex64",
   "C_RESIDUAL": 2.02142333984375,
   "C_OPERATOR": 7.500190734863281
  }
 },
 "caps": {
  "softmax": {
   "bytes_per_example": 135321.0,
   "n0": 52809,
   "n10": 47528,
   "n20": 42247,
   "pow2": 32768
  },
  "arm_pl": {
   "bytes_per_example": 135329.0,
   "n0": 52806,
   "n10": 47525,
   "n20": 42245,
   "pow2": 32768
  },
  "arm_phase": {
   "bytes_per_example": 258609.0,
   "n0": 27633,
   "n10": 24870,
   "n20": 22106,
   "pow2": 16384
  }
 },
 "arm_multiplier": {
  "softmax": 0.998597349564037,
  "arm_pl": 1.0415803530044996,
  "arm_phase": 2.0849676792867586
 },
 "contention_factor": 2.5462913927814568,
 "contention_factor_cuda": 1.046505922274206,
 "calibrate_takes_device": true,
 "recert_seconds": 0.7700957999986713,
 "determinism": {
  "off": {
   "softmax": {
    "secs": 0.010985550000441435,
    "grad_spread": 0.0
   },
   "arm_pl": {
    "secs": 0.011522350000632287,
    "grad_spread": 0.0
   },
   "arm_phase": {
    "secs": 0.02117050000015297,
    "grad_spread": NaN
   }
  },
  "strict": {
   "softmax": {
    "secs": 0.011722700000973418,
    "grad_spread": 0.0
   },
   "arm_pl": {
    "error": "RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation, but you set 'torch"
   },
   "arm_phase": {
    "error": "RuntimeError: cumsum_cuda_kernel does not have a deterministic implementation, but you set 'torch"
   }
  },
  "warn": {
   "softmax": {
    "secs": 0.012980599999536935,
    "grad_spread": 0.0
   },
   "arm_pl": {
    "secs": 0.013604750000013155,
    "grad_spread": 0.0
   },
   "arm_phase": {
    "secs": 0.02785829999993439,
    "grad_spread": NaN
   }
  }
 },
 "finite_gradient": {
  "softmax|as-constructed|cpu": null,
  "softmax|as-constructed|cuda": null,
  "arm_pl|as-constructed|cpu": null,
  "arm_pl|as-constructed|cuda": null,
  "arm_pl|identity_heads()|cpu": null,
  "arm_pl|identity_heads()|cuda": null,
  "arm_phase|as-constructed|cpu": 0,
  "arm_phase|as-constructed|cuda": 0,
  "arm_phase|identity_heads()|cpu": 14,
  "arm_phase|identity_heads()|cuda": 23
 },
 "determinism_boundary": {
  "torch.backends.cuda.matmul.allow_tf32": false,
  "torch.backends.cudnn.allow_tf32": true,
  "torch.get_float32_matmul_precision()": "highest",
  "tf32_rel_operator": 0.0003184080123901367,
  "tf32_rel_forward": 0.00030644419579189323,
  "cumsum": [
   {
    "reduce": 64,
    "batch": 1,
    "spread": 0.0
   },
   {
    "reduce": 10000,
    "batch": 1,
    "spread": 0.0
   },
   {
    "reduce": 1000000,
    "batch": 1,
    "spread": 2.2737367544323206e-12
   },
   {
    "reduce": 64,
    "batch": 2048,
    "spread": 0.0
   },
   {
    "reduce": 64,
    "batch": 32768,
    "spread": 0.0
   },
   {
    "reduce": 64,
    "batch": 262144,
    "spread": 0.0
   },
   {
    "reduce": 4096,
    "batch": 512,
    "spread": 0.0
   }
  ],
  "arm_gradient": [
   {
    "n": 2048,
    "spread": 0.0
   },
   {
    "n": 32768,
    "spread": 0.0
   }
  ]
 },
 "totals": {
  "150": {
   "cpu": 12.74808216822967,
   "cuda": 1.0680315455424365,
   "cells": 232,
   "cpu_pess": 32.46033189943398,
   "cuda_pess": 1.1177013375858331,
   "misfits": [
    [
     "10",
     "R2 BED-M t*=8",
     32768,
     [
      "arm_phase"
     ]
    ],
    [
     "17",
     "R3(a) delay bed",
     32768,
     [
      "arm_phase"
     ]
    ],
    [
     "18",
     "R3(b) power-law bed",
     32768,
     [
      "arm_phase"
     ]
    ],
    [
     "25",
     "R5 BED-1",
     32768,
     [
      "arm_phase"
     ]
    ]
   ]
  },
  "9600": {
   "cpu": 815.8772587666989,
   "cuda": 68.35401891471594,
   "cells": 232,
   "cpu_pess": 2077.461241563775,
   "cuda_pess": 71.53288560549332,
   "misfits": [
    [
     "10",
     "R2 BED-M t*=8",
     32768,
     [
      "arm_phase"
     ]
    ],
    [
     "17",
     "R3(a) delay bed",
     32768,
     [
      "arm_phase"
     ]
    ],
    [
     "18",
     "R3(b) power-law bed",
     32768,
     [
      "arm_phase"
     ]
    ],
    [
     "25",
     "R5 BED-1",
     32768,
     [
      "arm_phase"
     ]
    ]
   ]
  }
 }
}
```

---

## 8. LIMITS

Everything this node could not establish, collected once.

- **R1's 5.2×–5.8× is diagnosed to a mechanism, not to a magnitude.** The
  placement × contention cross reproduced 2.00×, 2.18×, 2.75×, 2.93×, 2.96× and
  4.03× on six independent measurements — 41–83% of the gap in log terms. The residual is
  not explained. What *is* established: the gap is not the arm, the shape, the
  data source, the thread count, the statistic or thermal drift, each
  eliminated by measurement in §2.2.
- **The exact machine state of `17:25`–`17:48` on 2026-08-31 is not
  recoverable.** §2.4's `lake build` at `17:43:35` is one dated competing
  process; how many others held the box is in no record this node can read.
- **`C_RESIDUAL` for the complex arm is NOT IDENTIFIED at this shape** (§3.2).
  The two-point solve returns 2.02, an artefact of the operator term being 84%
  of the total at `s = 64`, not a residual count. Every complex-arm projection
  here therefore uses measured bytes-per-example rather than the decomposition.
  If BED-K forces a longer sequence the decomposition must be re-identified at
  the new `s` before it can be trusted.
- **The composed arm's dtype is NOT MEASURED.** R3's composed arm does not
  exist. §6 prices it as complex, the pessimistic end; if X₃₈'s coupling is
  real-valued, R3(a) and R3(b) each fall by about 3.4 GPU-h and both fit at
  `n = 32768` (§7 section F's `NO FIT` flags clear on those two rows). This is the same NOT MEASURED the previous round filed, one arm
  further along.
- **`ceq/arm_phase.py` may not be it.5's arm.** Every `arm_phase` number here
  is measured against the module as it stands. it.5 builds "the phase-gate arm
  on §S-M′"; if that is a new or modified module, §3.3, §3.4, §5.3, §5.4 and
  every `arm_phase` hour in §6.2 must be re-measured with this probe before
  it.7 commits.
- **The determinism recommendation rests on four repeats at one shape.**
  `warn_only`'s gradient spread read `0.000e+00` for both real-valued arms, but
  four repeats inside one process is a weak test of cuBLAS algorithm selection.
  Re-measure at R2's shape before R2 is credited.
- **§5.2's items 4 and 5 are owed, not done.** This node priced the
  re-certification and deliberately did not perform it. The certificate in
  §6.3 is signed conditional on that work, and if the bar reads BROKEN on cuda
  the device decision reverses.
- **Five files were `M` in `git status` throughout.** §5 reads their state at
  run time rather than asserting it, but a certificate written against an
  uncommitted tree is a certificate about a moving object. HEAD and the dirty
  set are recorded at the top of this file.
- **The GPU lane's freedom from contention is a property of today, not a
  theorem.** `scale/hyperbolic.py:239` selects cuda automatically when it is
  available; a concurrent node running it would share the device and §2.6.3's
  schedulability argument would weaken by however much that takes.
- **VRAM free is a snapshot.** 6.939 GiB was free with 1.057 GiB held by the
  display. A graphical workload started mid-run reduces it. That is a nuisance
  at softmax's 38% headroom and it is why §3.4's caps are quoted at a 20%
  margin rather than at zero.
- **The 0.200 GiB eval reserve is `[ASSUMED]`**, a rounded-up bound rather than
  a measurement, carried unchanged from `V15_NEPTUNE_SYSTEMS.md` §5. Removing
  it moves softmax's `n_max` by about 1,600 and the phase arm's by about 830.
- **Arms-per-node in §6.1 is `[ASSUMED]` at every row.** The contract specifies
  its deciding measurements by their readings, not by their arm lists. A
  different arm census moves §6.2 proportionally; the two rows that dominate
  the total, R3(a) and R3(b), are the two whose arm list the contract states
  most explicitly, which is the right way round.
- **it.19 is priced but not scheduled** (§6.1). If D-CALIB-1 is honoured it
  belongs in the total, and at 28.08 GPU-h it is 41% of the plan.
- **The phase arm is not reachable from any runner.**
  `scripts/v15_r1.py`'s arm factory knows `arm_pl` and `m3_capability.Arm`
  only, and `scale/r10_capacity_sweep.py` constructs `Arm(str)` internally.
  Flagged in `V16_R1_DEVICE_READY.md` as still open. **Every `arm_phase` hour
  in §6.2 is priced for an arm no shipped runner can currently instantiate**;
  that is a precondition on it.5/it.7 rather than a systems finding, and it is
  named here because a schedule that assumes otherwise is wrong by the whole
  phase-arm column.
- **The determinism boundary is measured, not bounded.** §7's section I finds
  where `cumsum`'s run-to-run spread becomes nonzero at the arms' batched
  shapes. What it cannot do is prove it stays zero below that: 20 repeats on
  one card in one process is the same shape of claim `V16_BAR_RECERT.md` flags
  as insufficient about its own numbers, and this node inherits that limit
  rather than escaping it. §5.3.1 exit 2's claim is *"reproducible at these
  shapes over this many repeats"*, and it must be quoted with the repeat count
  attached or it becomes the assumption it replaced.
- **No arm was trained, fitted or scored.** Every timing and allocation used
  `torch.randn` or a corpus draw used for shape and bytes only. §5.4 ran
  gradient steps and reports **only whether the gradient was finite** — no
  loss, no NRMSE, no seed sweep, no verdict. Step cost and peak bytes are
  functions of tensor shape; gradient finiteness is not, which is exactly why
  §5.4 had to be measured rather than assumed.
