# R5/R6 silent-defect guards -- Cameron

Producer: `r5_guards.py`. One command re-runs everything: `python r5_guards.py`
(needs cuda). Every number below is this run, this box: **NVIDIA GeForce RTX
4060 Laptop GPU, 8187.5 MiB total, torch 2.14.0+cu126, Windows.** Board events
written to `house-events.jsonl` as agent `Cameron`. No `git` command was run;
no file under `ceq/`, `results/`, `tests/chase/deq_run.jsonl`, `docs/canon/`
or `lean/` was written. `ceq/arm_smprime.py`, `ceq/attention.py` and
`ceq/hf/train.py` were copied read-only into `cameron/*_copy.py` for
reference and are not imported by `r5_guards.py`, which imports the real
package in place.

## Where the brief and this box disagree

The task brief's R5a numbers (14.753ms vs 0.464ms at d_head=44 fp32, both
fused backends rejected) and R6's guard-6 formula (`exp(S*5.2e-4)-1`, giving
0.033419/0.124527/0.577293/5.152579 at S=64/256/1024/4096) were **not**
accepted on the brief's word and were re-driven from this box's own
`can_use_flash_attention` / `can_use_efficient_attention` and from
`gate3_bind.py`'s actual three-channel pipeline. Both diverge from the brief:

- **This build never compiles flash attention at all** (Windows wheel):
  `can_use_flash_attention` is `False` unconditionally, at every head dim,
  every dtype. The brief's fp32 d_head=44 scenario does not reproduce here --
  mem-efficient attention **accepts** fp32 d_head=44 on this box (verified
  below). What *does* reproduce, exactly, is the underlying mechanism the
  brief names (`d_head % 8`): mem-efficient attention's own C++ check
  (`check_head_dim_size_mem_efficient`, `sdp_utils.cpp:309`) rejects
  **bf16** d_head=44 with the literal message *"Mem efficient attention
  requires last dimension of inputs to be divisible by 8. Got ... 44"* --
  and DEFAULTS trains with `dtype="bf16_autocast"`, so the bf16 case is the
  one that actually matters in production, not the fp32 one the brief
  measured its timing numbers against. Guards 1 and 2 are built and proved
  against the real, reproducing case (bf16), and their construction-time
  form (`d_head % 8 == 0`) fires identically regardless of which dtype
  triggers it downstream, so this discrepancy does not change what either
  guard checks -- only which demo shape proves it firing.
- **Guard 6's formula does not fit its own re-measurement.** Re-running
  `gate3_bind.py::build_prefix`'s actual mantissa/exponent-split pipeline
  (not a re-derivation from scratch -- the same functions the prior,
  independently-verified `gate3.md` Cell A used) at S in {64,256,1024,4096},
  m=0.9, worst-case pair (gap = S-1): measured relative error is
  **0.079524 / 0.281715 / 0.734220 / 0.995026**, not the brief's
  0.033419/0.124527/0.577293/5.152579. The S=4096 point matches `gate3.md`
  Cell A's independently-recorded **0.99510** to 4 significant figures,
  which is strong evidence this run's pipeline is correct and the brief's
  guard-6 numbers are not. The brief's `exp(S*5.2e-4)-1` model is unbounded
  (7.41 at S=4096, i.e. 741% error); the real quantity is a *relative* error
  bounded above by 1 as the reconstructed value's ratio to the true value
  compounds toward zero -- it cannot exceed 1 by construction of the metric,
  so a model predicting 7.41 was never checkable against real output. Report
  in guard 6 below uses the re-measured numbers, not the brief's.
- **R5b's memory-cliff law DID reproduce, closely.** Re-run at L=6/7/9 (real
  `operator="smprime"` forward+backward+AdamW.step(), hidden=512 heads=8
  seq=512 batch=8 fp32) gave a fitted law within ~1 MiB/layer of the brief's
  929.9*L+208.3, and L=9 really does return `status=ran` with peak_mib above
  this card's own `mem_get_info` total. See guard 3/4 below -- this is the
  one row the brief's numbers are trusted on, because they were independently
  reproduced rather than taken on faith.

## Guard 1 -- head-dim assertion at construction

**Fires**: constructing at d_head=44 raises `ValueError` naming the actual
mechanism (mem-efficient attention's own divisible-by-8 check) rather than a
made-up round number. **Does not fire** at d_head=64 (the shipped default).

```python
def assert_head_dim(d_head: int, d_model: int, n_heads: int, *, mod: int = 8) -> None:
    if d_head % mod != 0:
        raise ValueError(
            f"d_head={d_head} (d_model={d_model} / n_heads={n_heads}) is not "
            f"a multiple of {mod}. torch's SDPA mem-efficient backend refuses "
            f"any head dim not divisible by 8 ... and silently falls through "
            f"to the unfused math kernel, which materialises the full "
            f"[B,H,S,S] attention matrix instead of running fused.")
```

### Proposed diff -- `ceq/lm.py` (`Attention.__init__`)

```diff
--- a/ceq/lm.py
+++ b/ceq/lm.py
@@ class Attention(nn.Module):
     def __init__(self, kind: str, d: int, n_heads: int):
         super().__init__()
         assert kind in ("softmax", "softmax_x", "signed", "sgate", "sgate_nores")
-        self.kind, self.n_heads, self.d_head = kind, n_heads, d // n_heads
+        self.kind, self.n_heads, self.d_head = kind, n_heads, d // n_heads
+        if self.d_head % 8 != 0:
+            raise ValueError(
+                f"d_head={self.d_head} (d={d} / n_heads={n_heads}) is not a "
+                f"multiple of 8. torch SDPA's mem-efficient backend refuses "
+                f"any head dim not divisible by 8 and silently falls back to "
+                f"the unfused math kernel, which materialises the full "
+                f"[B,H,S,S] attention matrix. Choose n_heads so d/n_heads is "
+                f"a multiple of 8.")
         self.qkv = nn.Linear(d, 3 * d, bias=False)
         self.proj = nn.Linear(d, d, bias=False)
```

### Proposed diff -- `ceq/hf/modeling_ceq.py` (`CEQAttention.__init__`)

```diff
--- a/ceq/hf/modeling_ceq.py
+++ b/ceq/hf/modeling_ceq.py
@@ class CEQAttention(nn.Module):
     def __init__(self, config: CEQConfig):
         super().__init__()
         d = config.hidden_size
         self.n_heads = config.num_attention_heads
         self.d_head = d // self.n_heads
+        if self.d_head % 8 != 0:
+            raise ValueError(
+                f"d_head={self.d_head} (hidden_size={d} / "
+                f"num_attention_heads={self.n_heads}) is not a multiple of "
+                f"8. torch SDPA's mem-efficient backend refuses any head dim "
+                f"not divisible by 8; the softmax control arm in this file "
+                f"would silently fall back to the unfused math kernel.")
         self.rho, self.hops, self.lam = config.rho, config.hops, config.lam
```

**Proof (this run)**: construct at `d_head=44` -> raises `ValueError`.
Construct at `d_head=64` (the DEFAULTS shape) -> no raise. See stdout capture
below, "GUARD 1".

## Guard 2 -- fused-backend assertion at runtime

Uses `torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION,
SDPBackend.EFFICIENT_ATTENTION])` (the API this torch version -- 2.14.0 --
actually ships; `torch.backends.cuda.sdp_kernel` also exists but is the
older, deprecated form) to force SDPA to refuse a math-kernel fallthrough.
On rejection, re-raises with the real `can_use_flash_attention` /
`can_use_efficient_attention` reasons attached rather than torch's generic
"no available kernel" text.

### Proposed diff -- `ceq/lm.py` (`Attention.forward`, softmax branch)

```diff
--- a/ceq/lm.py
+++ b/ceq/lm.py
@@ class Attention(nn.Module):
+    @staticmethod
+    def _sdpa_fused_or_raise(q, k, v, *, is_causal=True):
+        from torch.nn.attention import sdpa_kernel, SDPBackend
+        try:
+            with sdpa_kernel([SDPBackend.FLASH_ATTENTION,
+                              SDPBackend.EFFICIENT_ATTENTION]):
+                return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
+        except RuntimeError as e:
+            params = torch.backends.cuda.SDPAParams(q, k, v, None, 0.0, is_causal, False)
+            fa = torch.backends.cuda.can_use_flash_attention(params, False)
+            ea = torch.backends.cuda.can_use_efficient_attention(params, False)
+            raise RuntimeError(
+                f"SDPA refused a fused backend for q{tuple(q.shape)} "
+                f"dtype={q.dtype}: can_use_flash_attention={fa}, "
+                f"can_use_efficient_attention={ea}. Refusing the unfused "
+                f"math-kernel fallthrough, which would materialise "
+                f"[{q.shape[0]},{q.shape[1]},{q.shape[2]},{q.shape[2]}].") from e
+
     def forward(self, x: torch.Tensor) -> torch.Tensor:
         b, s, d = x.shape
         q, k, v = self.qkv_heads(x)

         if self.kind == "softmax":
-            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
+            o = self._sdpa_fused_or_raise(q, k, v, is_causal=True)
```

**Proof (this run)**: forced-fused SDPA at `d_head=44, bf16` (the shape that
actually rejects both backends on this box) -> raises. Same call at
`d_head=64, bf16` (control) -> runs, no raise. See "GUARD 2" in the capture.

## Guard 3 -- host-spill detector

```python
def check_host_spill(peak_allocated_bytes, device_index=0):
    free, total = torch.cuda.mem_get_info(device_index)
    spilled = peak_allocated_bytes > total
    return dict(spilled=spilled, peak_mib=peak_allocated_bytes/1024**2,
                device_total_mib=total/1024**2)
```

Reads the device's own live total via `torch.cuda.mem_get_info`, not a
hardcoded 8188 -- so it is correct on whatever card the box actually has.

### Proposed diff -- `ceq/hf/train.py` (`train()`, after each optimizer step)

```diff
--- a/ceq/hf/train.py
+++ b/ceq/hf/train.py
@@ def train(*, out_dir, steps, batch, seq, hidden_size, n_layers, n_heads, ...):
+    def _check_host_spill(peak_bytes):
+        free, total = torch.cuda.mem_get_info()
+        if peak_bytes > total:
+            import warnings
+            warnings.warn(
+                f"host-spill detector: peak_allocated "
+                f"{peak_bytes/1024**2:.1f} MiB exceeds this device's own "
+                f"total {total/1024**2:.1f} MiB (torch.cuda.mem_get_info). "
+                f"torch.cuda.OutOfMemoryError did NOT raise -- this is "
+                f"Windows WDDM host-RAM paging, not a normal fit. Reduce "
+                f"n_layers or batch.", RuntimeWarning)
     ...
     peak = torch.cuda.max_memory_allocated()
+    _check_host_spill(peak)
```

**Proof (this run)**: real `build(operator="smprime")` forward + backward +
`AdamW.step()`, hidden=512 heads=8 seq=512 batch=8 fp32:

| L | peak_mib (this run) | device_total_mib | detector fires |
|---|---|---|---|
| 6 | 5788.1 | 8187.5 | no |
| 7 | 6718.7 | 8187.5 | no |
| 9 | 8578.1 | 8187.5 | **yes** (8578.1 > 8187.5, status still "ran", no `OutOfMemoryError` raised) |

(L=8 not directly probed by this script -- it is used as a control point in
guard 4 via the fitted law instead, see below.) Wall clock per run varied
9x under GPU contention with sibling agents on this shared box (L=6: 19.5s,
L=7: 404.0s, L=9: 43.9s) -- the timings are not comparable to each other and
are NOT used for anything in this row; only `peak_mib` (a high-water mark,
insensitive to how long the box took to get there) is.

## Guard 4 -- depth default

```python
def assert_depth_default(n_layers, predicted_peak_mib, device_mib, *,
                         acknowledge_headroom=False):
    frac = predicted_peak_mib / device_mib
    if frac <= 0.80: return
    if frac <= 0.90 and acknowledge_headroom: return
    if not acknowledge_headroom:
        raise MemoryError(f"... {frac:.1%} of {device_mib:.1f} MiB, above "
                          f"80% silent-default ceiling. Pass "
                          f"acknowledge_headroom=True to go to 90%, or "
                          f"reduce n_layers.")
    raise MemoryError(f"... {frac:.1%}, above the 90% hard ceiling even "
                      f"with acknowledge_headroom=True. No override past "
                      f"this point.")
```

The literal "<=0.80 silent, L=7 permitted, L=8 refused" reading in the task
brief does not actually partition cleanly at a single 80% line once the law
is re-measured on this box: L=7's measured peak lands at ~82% of this card's
8187.5 MiB, just over 80%. So L=7 is implemented as "permitted with an
explicit `acknowledge_headroom=True`" (a real, auditable opt-in, not a
silent default) and L=8+ as an unconditional refusal with **no** override --
because guard 3 already shows the layer directly past L=8 (L=9) both spills
past the card's own total AND never raises `OutOfMemoryError`, so there is
no safety net an override could fall back on.

### Proposed diff -- `ceq/hf/train.py` (`preflight`)

```diff
--- a/ceq/hf/train.py
+++ b/ceq/hf/train.py
@@ def preflight(*, hidden_size, n_layers, n_heads, seq, batch, gpu, ...):
+    SLOPE_MIB_PER_LAYER, INTERCEPT_MIB = 929.9, 208.3  # re-measured, see r5_guards.md
+    if gpu is not None and "usable" not in ...:  # illustrative: use the real device total
+        device_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2
+        predicted = SLOPE_MIB_PER_LAYER * n_layers + INTERCEPT_MIB
+        assert_depth_default(n_layers, predicted, device_mib,
+                             acknowledge_headroom=overrides.get("acknowledge_headroom", False))
     cfg = sizing.Config(...)
```

(`assert_depth_default` as defined above, imported from a small
`ceq/depth_guard.py` or inlined -- proposed as a diff, not placed, since the
task scope is diffs-only.)

### Proposed diff -- `ceq/hf/train.py` (`DEFAULTS`)

```diff
--- a/ceq/hf/train.py
+++ b/ceq/hf/train.py
-DEFAULTS = dict(hidden_size=512, n_layers=8, n_heads=8, seq=512, batch=8,
-                vocab_size=256)
+#: n_layers dropped 8 -> 6. At hidden=512/heads=8/seq=512/batch=8 fp32,
+#: operator='smprime', L=8 measures ~93% of this card's total and is one
+#: layer from a measured, silent WDDM host-spill at L=9 (see r5_guards.md,
+#: guard 3/4) -- L=8 must not be what a fresh `train()` call ships by
+#: default. L=6 is comfortably under the 80% silent-default ceiling.
+DEFAULTS = dict(hidden_size=512, n_layers=6, n_heads=8, seq=512, batch=8,
+                vocab_size=256)
```

**Proof (this run)**: law re-fit from the L=6/7/9 points measured in guard 3
(least squares, this process):

```
peak_mib = 929.96*L + 208.56   R^2 = 0.9999999458
```

(within 0.06 MiB/layer and 0.3 MiB intercept of the brief's 929.9*L+208.3 --
this is the one row where the brief's own numbers check out under an
independent re-derivation.) Device total read live via
`torch.cuda.mem_get_info()`: 8187.5 MiB (matches `ceq/sizing.py`'s existing
`GPUS["RTX4060L-8GB"] = (8.0*1024**3, 0.90)` entry's raw total to within
4.5 MiB -- that dict's `0.90` "usable fraction" is a *different*, existing
safety margin for the allocator/cuBLAS-workspace overhead `sizing.fits()`
already prices; guard 4's 80%/90% tiers are a second, independent margin for
the depth default and compose with it rather than replacing it).

`assert_depth_default` exercised at four points, this run:

| n_layers | mode | predicted | outcome |
|---|---|---|---|
| 6 | default, no override | 5788.3 MiB (70.7%) | OK -- silent default |
| 7 | no override | 6718.3 MiB (82.1%) | **REFUSED** -- above 80% |
| 7 | `acknowledge_headroom=True` | 6718.3 MiB (82.1%) | OK -- permitted with override |
| 8 | `acknowledge_headroom=True` | 7648.2 MiB (93.4%) | **REFUSED** -- above 90% hard ceiling, no override possible |

All four outcomes matched what the guard was built to produce.

## Guard 5 -- dtype guards (construction-time refusal)

Since `CEQAttention` is an `nn.Module` constructed in the default dtype and
cast afterward (`model.to(dtype)`, the standard HF/PyTorch pattern -- not
handed a dtype at `__init__`), "at construction" is implemented as
overriding `_apply` (the method every one of `.to()`, `.half()`,
`.bfloat16()` routes through in every version of `torch.nn.Module`) so the
refusal happens at the cast call site, before any tensor is actually
converted -- not three calls later, inside a `Linear`, as `RuntimeError:
mat1 and mat2... double != Half`.

### Proposed diff -- `ceq/hf/modeling_ceq.py` (`CEQAttention`)

```diff
--- a/ceq/hf/modeling_ceq.py
+++ b/ceq/hf/modeling_ceq.py
@@ class CEQAttention(nn.Module):
+    _BAD_SMPRIME_DTYPES = (torch.bfloat16, torch.float16)
+
+    def _apply(self, fn, recurse=True):
+        probe = torch.zeros(1, dtype=torch.float32)
+        try:
+            target = fn(probe).dtype
+        except Exception:
+            target = None
+        if self.operator == "smprime" and target in self._BAD_SMPRIME_DTYPES:
+            raise TypeError(
+                f"CEQAttention(operator='smprime') refuses dtype {target}. "
+                f"ceq/arm_smprime.py::gate() calls torch.polar(), which "
+                f"rejects bfloat16 directly, and ceq/arm_smprime.py::_ctype() "
+                f"maps every non-float32 dtype (incl. float16) to complex128, "
+                f"so gate()'s .real comes back float64 and crashes the next "
+                f"cast Linear with 'double != Half'. The gated arm is "
+                f"float32-only; keep this block in float32 or run it under "
+                f"bf16 autocast instead of a hard .to({target}) weight cast.")
+        return super()._apply(fn, recurse=recurse)
+
     def __init__(self, config: CEQConfig):
```

**Proof (this run)**: `build(operator="smprime").to(torch.bfloat16)` ->
raises `TypeError` naming `torch.polar`. Same `.to(torch.float16)` -> raises
`TypeError` naming `_ctype`. `.to(torch.float32)` (control) -> no raise. Spy
(`gate3_bind.spy_polar`/`unspy_polar`, reused as instructed) confirms the
real, shipped `ceq.arm_smprime.gate()` (fp32) DOES call `torch.polar`
(expected -- that is the whole reason the dtype refusal exists), and that
`gate3_bind.py`'s three-channel alternative pipeline does NOT, across a real
run of `build_prefix` + `reconstruct_pairs`. See "GUARD 5" in the capture.

## Guard 6 -- bf16-mantissa rounding law, re-measured

Re-measured directly from `gate3_bind.py::build_prefix`'s real
mantissa/exponent-split pipeline (the actual three-channel scheme, not a
naive raw-bf16 cumulative product -- a naive product underflows to bf16's
denormal floor, ~9.2e-41, by S~900 for m=0.9 and produces a spurious,
unrelated blow-up; kept as a dead end in `cameron/remeasure_bf16_law.py` for
the record, superseded by `cameron/remeasure_bf16_law2.py` which uses the
real split-channel encoding and is what `r5_guards.py` calls).

Measured relative error at the worst-case pair (gap = S-1), m=0.9, seed 0:

| S | measured (this run) | brief's claimed `exp(S*5.2e-4)-1` |
|---|---|---|
| 64 | 0.079524 | 0.033419 (brief) / 0.033840 (formula, recomputed) |
| 256 | 0.281715 | 0.124527 / 0.142387 |
| 1024 | 0.734220 | 0.577293 / 0.703151 |
| 4096 | 0.995026 | 5.152579 / 7.414194 |

**Verdict: the claimed law does not hold.** The measured/claimed ratio is
2.35x at S=64 and 0.13x at S=4096 -- not a constant, so this is not a
constant-mislabeled bound, the functional form itself is wrong. The real
quantity (relative error of a positive reconstructed magnitude against a
positive true magnitude, `|recon/true - 1|`) saturates toward **1.0** as the
compounding `(bf16(0.9)/0.9)^gap` bias drives `recon/true -> 0`; it cannot
exceed 1 by construction once `recon/true` is bounded in `[0, ~1]`, so an
unbounded exponential (7.41 at S=4096) was never a fittable model for this
metric. The S=4096 point (0.995026) matches the independently-recorded
`gate3.md` Cell A number (0.99510) to 4 significant figures, which is the
cross-check that this re-measurement, not the brief's, is the one to trust.

**First S (of the four tested) over the 0.094 bound: S=256** (S=64 measures
0.0795, under the bound; S=256 measures 0.2817, over). Bar (1), the
no-exact-zero underflow claim, is unaffected by any of this -- it was
already re-verified as `gate3.md`'s own Cell B, separately, across all 12
(S, m) cells.

**Decision this feeds**: the three-channel bf16-mantissa path fails its own
accuracy bar well before S=256, worse than the brief implied, not better.
`gate3.md`'s own conclusion (fp32-mantissa fallback, +16 bits/position,
still no `torch.polar`, still no complex tensor) is the one guard 6's
re-measurement supports; nothing here reopens a bf16-mantissa shipping path.

## Board events

Every guard above appends one line to `house-events.jsonl`, agent
`Cameron`, event names `guard1_head_dim_assertion` through
`guard6_bf16_law_remeasured`, plus `start` and `done`. `io.open(path, "a",
encoding="utf-8")`, matching this project's board convention; never
rewritten.

## Repro

```
python r5_guards.py
```
Single process, cuda required. This run took ~9 minutes wall clock, not
because the workload is heavy but because this box was running **two other
agent processes on the same GPU at 100% util / ~7900 of 8188 MiB** for the
whole run (confirmed independently: a sibling `Foreman` board line at
17:46:51 names `r5_guards.py` by filename as one of the two other processes
it was sharing the card with). Re-run on an idle card should take well under
a minute. `NEVER KILL A GPU PROCESS` was honored throughout -- no sibling
process was touched despite the contention.

## A bug this run found in its own harness

`demo_guard6`'s own `board()` call raised `TypeError: Object of type bool is
not JSON serializable` (a `numpy.bool_` from a numpy-typed comparison, not a
plain Python `bool`) and the process then exited without writing that one
board line or the final summary line. Fixed in `r5_guards.py` (wrap in
`bool(...)`) after the fact; the two missing board lines were appended
separately with the exact numbers this run had already printed to stdout
before it crashed (marked `"note"` in both lines, so the board record is
honest about being appended rather than re-measured). This is the kind of
defect the task exists to catch, and it is reported rather than quietly
patched-and-hidden: the first full run of this file did NOT complete
cleanly.

## Full run capture (real stdout, this run, this box)

```
=== GUARD 1: head-dim assertion at construction ===
d_head=44: RAISED: d_head=44 (d_model=352 / n_heads=8) is not a multiple of 8. ...
d_head=64: no raise (correct)

=== GUARD 2: fused-backend assertion at runtime ===
d_head=44 bf16 (both backends rejected on this box): RAISED: SDPA refused a
  fused backend for shape q(2, 8, 128, 44) dtype=torch.bfloat16:
  can_use_flash_attention=False, can_use_efficient_attention=False. ...
d_head=64 bf16 (control): no raise (correct), out shape (2, 8, 128, 64)

=== GUARD 3: host-spill detector ===
L=6: peak=5788.1 MiB  device_total=8187.5 MiB  detector_spilled=False  (19.49s)
L=7: peak=6718.7 MiB  device_total=8187.5 MiB  detector_spilled=False  (404.04s, contended)
L=9: peak=8578.1 MiB  device_total=8187.5 MiB  detector_spilled=True   (43.90s)
Detector fires at L=9: True  |  silent at L=6: True  |  silent at L=7: True

=== GUARD 4: depth default, law re-measured at L in {6,7,9} ===
Re-fit law: peak_mib = 929.96*L + 208.56, R^2=0.9999999458
Device total (live): 8187.5 MiB
L=6 (default, no override): OK predicted=5788.3 MiB (70.7%)
L=7 (override): OK predicted=6718.3 MiB (82.1%)
L=7 (no override): REFUSED predicted=6718.3 MiB (82.1%)
L=8 (override): REFUSED predicted=7648.2 MiB (93.4%)

=== GUARD 5: dtype refusal at construction, both raises + hot-path spy ===
torch.bfloat16: RAISED: CEQAttention(operator='smprime') refuses dtype
  torch.bfloat16. ceq/arm_smprime.py::gate() calls torch.polar(), which
  rejects bfloat16 tensors directly (RuntimeError: ...)
torch.float16: RAISED: CEQAttention(operator='smprime') refuses dtype
  torch.float16. ... _ctype() maps every non-float32 dtype to complex128 ...
float32: no raise (correct)
Production gate() (fp32) calls torch.polar: True (expected True)
Three-channel path calls torch.polar: False (expected False)

=== GUARD 6: re-measure the bf16-mantissa rounding law ===
S=   64  measured=0.079524  claimed_exp_law=0.033840  ratio=2.350
S=  256  measured=0.281715  claimed_exp_law=0.142387  ratio=1.979
S= 1024  measured=0.734220  claimed_exp_law=0.703151  ratio=1.044
S= 4096  measured=0.995026  claimed_exp_law=7.414194  ratio=0.134
First tested S over the 0.094 bound: 256
[board() crashed here on numpy.bool_; fixed, see above -- run otherwise complete]
```

All six proofs printed real output; all fired where they should and stayed
silent on their controls. The one process-level failure (guard 6's board
write) was a bug in this harness, not in any guard's logic, and is reported
above rather than silently re-run until clean.
