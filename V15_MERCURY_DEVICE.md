# V15 MERCURY — `--device` FOR `scale/r10_capacity_sweep.py`

Required predecessor to it.9, per `V15_NEPTUNE_SYSTEMS.md` §6.3: a `--device`
flag threaded through `scale/r10_capacity_sweep.py`, with the three attached
conditions. Repo at the commit this node started from; box is the RTX 4060
Laptop GPU `V15_NEPTUNE_SYSTEMS.md` §1 measured.

---

## (a) What was added, and its default

`scale/r10_capacity_sweep.py`:

```
ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"), ...)
```

**Default `"cpu"` preserves current behaviour exactly.** Verified rather than
assumed: before this change the file called no `.to(...)` and passed no
`device=` anywhere, so every published number was measured on whatever device
`torch.randn`/`nn.Linear` land on with no argument — cpu. With `--device cpu`
(the default), `device = None` throughout, `train_with_checkpoints` never
calls `.to()`, and `batch_fn(..., device=None)` resolves to
`torch.device("cpu")` inside `negation_scope.make_equilibrium_batch` exactly
as it always did. `tests/loop` passing at the same 15 failures (below) is the
check that this claim holds outside this file's own reasoning too.

Threading: `batch_fn(..., device=device)` for `x_eval`/`y_eval` and each
`x_tr`/`y_tr` (the M3 batch builders already accept `device=`, unused by this
file until now); `train_with_checkpoints(..., device=device)` moves only the
model, and moves it *after* construction (`torch.manual_seed(seed); model =
Arm(arm, s)` first, `.to(device)` second) so initial weights are drawn from
the seeded CPU generator on every device — `.to()` copies values, it does not
redraw them. This mirrors `scale/paired_arm.py:41-66`'s existing
cpu/cuda-cell pattern, which is the closest sibling in the tree and was
followed rather than re-derived.

`GATE_TOL = 1e-3` (module constant, unchanged) and the rest of the file's
structure were not restructured.

---

## (b) The three conditions

1. **NO POOLING ACROSS DEVICES — DONE.** `device` is now written into every
   `header`/`cell`/`instrument_broken`/`dropped` journal record
   (`scale/r10_capacity_sweep.py`), and `refuse_cross_device_pool()` raises
   unconditionally on any pooled set spanning more than one device.
   Must-fire test at `tests/loop/test_no_cross_device_pooling.py` (4 tests,
   all passing, including the must-fire and its single-device control).
   `scale/identity_manifest.py` was **not modified**: its `CONFIG_FIELDS`
   already carries `"device"` as a first-class field (line 69), landed by
   another iteration before this node started — confirmed by reading the
   file rather than assumed, so no write was needed there.

2. **RE-CERTIFY ON DEVICE — BLOCKED, and `--device cuda` refuses to score.**
   `negation_scope.calibrate_bar` takes no `device` argument: its reference
   batch (`(batch_fn or make_batch)(n, s, d, seed=seed)`, no `device=`), its
   positive-control net, and its `feats` tensor are built on cpu on every
   call, cuda run or not. There is no flag that makes it measure anything
   about cuda; making it device-aware needs an edit to
   `scale/negation_scope.py`, which is outside this node's write scope (owned
   by another node this round, per the hard constraints). `GATE_TOL = 1e-3`
   is likewise a threshold measured over 16 untrained **CPU** seeds (see its
   docstring in `train_with_checkpoints`). Rather than score a cuda cell
   against either silently, `main()` now aborts before opening a journal when
   `--device cuda` is passed, printing the reason and returning 1 — see (e).

3. **DECLARE NONDETERMINISM — DONE, and it is free at this shape.**
   `torch.use_deterministic_algorithms(True)` (plus
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`, set before any cuda call) is enabled
   unconditionally on the `--device cuda` path; `torch.backends.cudnn.
   deterministic = True` is set too, matching `paired_arm.py`'s existing
   lane (this Arm has no cudnn-backed op, so that particular flag is inert
   here, but is set for consistency with the sibling file). Measured cost,
   instrumentation only, not journalled as a reading: 20 timed steps at
   `s=64, n=2048, t*=8, softmax`, warm-up discarded —
   **`deterministic=False`: 0.01120 s/step; `deterministic=True`: 0.01031
   s/step; ratio 0.921x**, i.e. no slowdown, inside run-to-run noise. Every
   header record now also carries `deterministic_algorithms:
   torch.are_deterministic_algorithms_enabled()`, so the setting is read off
   live torch state rather than assumed.

---

## (c) `tests/loop` before/after

| | failed | passed |
|---|---|---|
| baseline (`results/v15_loop_suite5.txt`) | 15 | 501 |
| after this change | **15** | **507** |

The 15 failing test names are **identical**, line for line, between the
baseline file and this run (diffed directly, not eyeballed):
`test_conftest_import_is_order_dependent.py` x10 (one per file),
`test_corpus_is_recoverable_and_verifiable.py` x2,
`test_every_boundary_node_can_propagate.py` x1,
`test_manifest_refuses_an_absence_it_has_not_earned.py` x1,
`test_the_bar_control_is_scored_out_of_sample.py` x1. None of these touch
`scale/r10_capacity_sweep.py`, `device`, or pooling. **No new failure.**

Passed count moved 501 → 507, +6: the four tests in
`tests/loop/test_no_cross_device_pooling.py`, plus two additional
parametrized nodes picked up by repo-wide scanning tests (e.g. the
import-order and no-import-time-write meta-tests, which parametrize over
every `.py` file in the tree) once that new file existed — not new logic
under test, just those scans enumerating one more file cleanly.

Command: `python -m pytest tests/loop -q --no-header -p no:cacheprovider`.

---

## (d) CPU vs CUDA, one cell, side by side

Instrumentation only — `t*=2, n_train=512, n_eval=512, seed=0, arm=softmax,
steps=150`, called directly through `train_with_checkpoints` (bypassing
`main()`'s CLI, which is what condition 2 gates — see (e)). Not written to
`results/`, not a capability reading: this `n`/step count is far below
anything the it.9 plan would credit, chosen only to be cheap on both devices.
Both cells share model-init weights (constructed on the seeded CPU generator
before either copy is placed) and share input tensor *bytes* (both corpora
are drawn on the CPU generator inside `make_equilibrium_batch` and only then
moved — `negation_scope.py:94-96`), so any reading difference below is device
arithmetic, not a different draw.

| | CPU | CUDA | \|diff\| |
|---|---|---|---|
| `red.nrmse0_train` | 1.0029066313178907 | 1.0029066313178907 | **0** (bit-identical) |
| `red.nrmse0_eval` | 0.9993243299625522 | 0.9993243299625522 | **0** (bit-identical) |
| `train_nrmse` (after 150 steps) | 0.5562783644490661 | 0.5562920624050564 | 1.36980e-05 |
| `eval_nrmse` (after 150 steps) | 1.3152986427866267 | 1.3153114805975745 | 1.28378e-05 |
| `secs` | 4.18 | 0.46 | 9.1x |

**Manifests** (`scale.identity_manifest.manifest()`, read-only, used here
only for this comparison — not wired into the shipped path):

```
CPU   values: {kind: softmax, task: e3_t2, s: 64, d: 24, d_model: 16, steps: 150,
              n_train: 512, n_eval: 512, seed: 0, device: cpu,  torch_version: 2.5.1+cu121}
      hash:   593e85a4e4c305fca128c85b9058207fbcfc9564e127687b32455289ee27c926

CUDA  values: {kind: softmax, task: e3_t2, s: 64, d: 24, d_model: 16, steps: 150,
              n_train: 512, n_eval: 512, seed: 0, device: cuda, torch_version: 2.5.1+cu121}
      hash:   baa1e32676dfa3d4598204f074f2867882e2398b4c89cd823dc04cd20e070f60
```

Hashes differ, as they must (`device` is a `CONFIG_FIELDS` entry); both list
the same `absent = [cell, k_piv, beta, t_max, n_neumann]` (fields this Arm
does not have — QuintArm-only fields).

**The difference, and its size.** The 0-step RED gate read bit-identical at
this shape/seed — a genuine measurement, not an assumption, but a single
(tiny `n`, single seed) data point, not a certification that `GATE_TOL=1e-3`
holds on cuda in general (that is exactly what condition 2 is refusing to
claim). After 150 trained steps the two readings separate by ~1.3e-5 in
`eval_nrmse` — real, and the direction cuBLAS's non-bitwise reduction order
predicts, but **two orders of magnitude smaller** than M-10's measured
thread-count effect (2.345e-3) at this tiny cell. That does not weaken
condition 1: this comparison ran 150 steps at `n=512`; the deciding cells run
up to 9600 steps at `n=32768`, and nothing here measures whether the gap
compounds with more optimizer steps on a non-bitwise-reduced device. The
guard in (b.1) does not depend on the gap being large — it refuses on any
recorded device difference, which is what condition 1 asks for.

Command used (instrumentation script, scratchpad, not committed):
`train_with_checkpoints` called directly for both devices; full script
available on request but not part of this node's write scope deliverables.

---

## (e) Can `--device cuda` currently produce a scored verdict?

**No.** `main()` aborts immediately after argument parsing, before opening a
journal file, printing:

```
ABORT: --device cuda cannot produce a scored verdict here.
  calibrate_bar() and GATE_TOL=1e-03 were established on cpu and neither
  re-certifies on cuda without editing scale/negation_scope.py, which is
  outside this node's write scope. See V15_NEPTUNE_SYSTEMS.md condition 2 and
  MISTAKES.md V-22.
```

and returns exit code 1. The mechanism condition 2 needs to lift this
(`train_with_checkpoints(..., device=...)`, `refuse_cross_device_pool`,
device threaded to every batch call) is fully wired and tested; only the
credited path is gated, on a re-certification this node cannot perform inside
its write scope. Lifting the abort is a one-line deletion once
`scale/negation_scope.py::calibrate_bar` accepts and honours a `device`
argument and has been re-run on cuda by whichever node owns that file.

---

## Limits

- The CPU-vs-CUDA comparison in (d) is one seed, one tiny cell. It is not a
  substitute for condition 2's 16-seed re-certification of `calibrate_bar`
  and `GATE_TOL` on cuda, and is not offered as one.
- `torch.backends.cudnn.deterministic = True` is set for parity with
  `paired_arm.py`'s cuda lane but has no effect on this Arm (no cudnn-backed
  op in `Arm.forward`); the operative determinism control measured and set
  here is `torch.use_deterministic_algorithms(True)`.
- `refuse_cross_device_pool` is a new function in `scale/r10_capacity_sweep.py`,
  not wired into `scale/it11_verdict.py::by_seed` (out of write scope). A
  consumer that pools rows via `by_seed` alone, without also calling
  `refuse_cross_device_pool`, is not protected by this change — it is
  protected only accidentally, the way it always was, by `by_seed`'s
  same-seed-same-threads value check.
- Whether the ~1.3e-5 device gap at 150 steps / n=512 grows at the deciding
  cells' scale (9600 steps, n=32768) is NOT MEASURED here — flagged, not
  filled in, per "nothing trains as a research reading."
