# V17-K Gate-0: G0.2 K-RESUME and G0.3 K-PERSIST

**LEAD CAVEAT.** Everything below is an INSTRUMENT CHECK on the checkpoint path.
Nothing here is a research reading: no NRMSE, no cell of R1′/R2/any bed, no
verdict on the arm. Runs of `train()` at 11,488 parameters and one at 25,728,000
appear in this document and none of them is a result about the operator. The bitwise equalities are measured on **CPU** at a tiny shape,
because bitwise equality is not a performance question and the one 8 GB card on
this box is shared by five agents; one CUDA confirmation at the same tiny shape
is reported and flagged for the contention it cost. Nothing was run on Kaggle,
the kaggle CLI was not invoked, and `~/.kaggle` was not touched.

Repository state, both ends of the work:

| | `git rev-parse HEAD` | `git status --porcelain` |
|---|---|---|
| start | `ab5b48547884e04258276e6e808d5a71ea65f917` | `M scripts/v15_r1.py` |
| end | `ab5b48547884e04258276e6e808d5a71ea65f917` | `M ceq/arm_smprime.py`, `M ceq/hf/configuration_ceq.py`, `M ceq/hf/modeling_ceq.py`, `M ceq/hf/train.py`; untracked `V17_G01_IDENTITY.md`, `V17_G02_G03_CHECKPOINT.md`, `V17_G04_ENV.md`, `V17_G05_DATA.md`, `V17_G08_G09_AUTOPILOT.md`, `V17_LABEL_CELL_REPAIR.md`, `V17_NOTEBOOK.md`, `ceq/autopilot.py`, `ceq/kdata.py`, `kaggle/`, `requirements-kaggle.txt`, `results/k_cert_local.json`, `results/k_data_manifest.json`, `scripts/k_cert.py`, `tests/gate0/` |

No writing git command was run. Of the four modified files, **only
`ceq/hf/train.py` belongs to G0.2/G0.3** (158 insertions, 22 deletions); `ceq/arm_smprime.py`,
`ceq/hf/configuration_ceq.py` and `ceq/hf/modeling_ceq.py` were changed by other
agents during this work and are on this item's do-not-edit list. `scripts/v15_r1.py` was
modified at the start of this session and clean at the end; that change was not
made or reverted here. Of the untracked entries, `V17_G02_G03_CHECKPOINT.md`, the two
`tests/gate0/test_g0[23]_*.py` files, `tests/gate0/helpers.py` and
`tests/gate0/conftest.py` belong to this item; the rest belong to the four
agents working this tree in parallel. The final test runs below were made with their
`modeling_ceq.py` edits in place.

---

## 1. Verdict

| item | call | what carries it |
|---|---|---|
| **G0.2 K-RESUME** | **GREEN** | bitwise on all four components, k=2 → k+m=5, CPU and CUDA; three planted negatives, two fired on weights, one fired on its own component only (§5) |
| **G0.3 K-PERSIST** | **GREEN** | atomic write RED→GREEN, both kill-mid-write modes leave the last good checkpoint byte-identical and still resuming bitwise; **periodic checkpoints** with a mid-chunk resume proved bitwise (§7.6); rotation sized at 64 checkpoints per 20 GB, policy holds 4 at 6.18 % (§8) |

`tests/chase/test_resume_checkpoint.py`: **3 passed before, 3 passed after**
(11.05 s → 13.30 s). Not edited. Together with `tests/chase/test_colab_chain.py`,
the other chase file that drives `train()`: **25 passed** on the final run.
`tests/gate0/test_g02_resume.py` + `tests/gate0/test_g03_persist.py`:
**25 passed** (10 + 15). All four files together on the final run:
**50 passed, 67.71 s**. `tests/loop`: **15 failed, 525 passed** — the standing
baseline, with no `tests/gate0` node ID in it (§5).

---

## 2. What was already covered, and what was not

`ceq/hf/train.py::train()` before this work already wrote
`{out_dir}/trainer_state.pt` with `optimizer`, `step`, `torch_rng_state` and
`data_gen_state`, and already took `resume_from`.
`tests/chase/test_resume_checkpoint.py` (3 tests) already covered:

- the stitched loss sequence against an uninterrupted run, within `1e-5` relative;
- the **model parameters** bitwise, via `torch.equal`;
- one planted negative — a no-op'd `AdamW.load_state_dict` must change the weights.

**Not covered anywhere before this work**, and now covered by
`tests/gate0/test_g02_resume.py` and `tests/gate0/test_g03_persist.py`:

| gap | now |
|---|---|
| optimizer state compared bitwise (`exp_avg`, `exp_avg_sq`, `step`) | §4 |
| torch RNG state compared bitwise | §4 |
| data-generator state compared bitwise | §4 |
| the docstring's claim that the generator state IS the dataloader position | §6 — **measured**, not argued |
| non-degeneracy of the comparator (1 ulp) | §3 |
| planted negative: RNG state not restored | §5 |
| planted negative: data-generator state not restored | §5 |
| atomic checkpoint write | §7 |
| kill mid-write, kill between write and rename | §7 |
| rotation sized to `/kaggle/working` | §8 |
| resume into the directory being resumed from | §7.5 |
| periodic (mid-chunk) checkpoints, and resume from one | §7.6 |

`train()`'s docstring argued the dataloader case rather than measuring it:
*"batches are drawn by index from `gen`, so the generator's state IS the
dataloader position"*. §6 measures it.

---

## 3. The comparator, and its non-degeneracy check

`tests/gate0/helpers.py::bitwise_diff(a, b)` walks two nested structures and
returns every path at which they differ. Leaves are compared with `torch.equal`
— never `allclose`, never a tolerance. It descends into dicts and lists because
only one of the four components is a flat dict of tensors: the optimizer's
`state_dict()` is `{"state": {i: {"exp_avg", "exp_avg_sq", "step"}},
"param_groups": [...]}`, and a comparison that stopped at the top of that would
report equality while the moments drifted.

`load_state(dir)` reads all four components **back off disk**, not from live
objects a run left in memory. What survives a session cap is the directory.

**Non-degeneracy — the comparator catches one ulp.** This repo has struck 14
vacuous controls, so the PASS in §4 is void unless the comparator is shown to
detect the smallest difference the dtype admits. All three calibrations call the
**same** `bitwise_diff` the PASS half calls, on the **same** `load_state` output,
and each asserts the diff list has length **exactly 1** — so the comparator is
shown both to see the change and not to over-report:

| calibration | perturbation | result |
|---|---|---|
| `test_comparator_catches_a_one_ulp_perturbation` | `torch.nextafter` on one element of `model.embed_tokens.weight`, asserted `≤ 2⁻²³` relative | 1 path reported |
| `..._in_the_optimizer` | `torch.nextafter` on one element of `optimizer.state.0.exp_avg` | 1 path reported |
| `..._one_byte_flip_in_the_rng_state` | +1 on byte 0 of each uint8 buffer | 1 path reported, each |

Two further anti-vacuity guards:
`test_each_component_is_actually_present_and_nonempty` pins 15 parameters, 15
optimizer state entries with exactly `{exp_avg, exp_avg_sq, step}`, and
5,056-byte RNG buffers — so none of the four names resolves to nothing; and
`bitwise_diff` **reports** an empty-dict-vs-empty-dict comparison as
`[VACUOUS: empty dict on both sides]` rather than passing it.
`test_no_resume_at_all_differs` is the floor control: a run that never resumed
must not match the reference.

---

## 4. G0.2 — bitwise on all four components

k = **2**, m = **3**, deliberately unequal: at k == m an optimizer whose `step`
counter were reset on resume would still land on the right total and the
comparison would not notice. Shape hidden 16 / L1 / H2 / seq 8 / batch 2,
vocab 256 = 11,488 parameters — the shape
`tests/chase/test_resume_checkpoint.py:21` already trains **[INHERITED**, cited
by file**]**. Device **cpu**, `torch.set_num_threads(1)`.

Uninterrupted 5 steps into `ref`; 2 steps into `a`; 3 steps into `b` with
`resume_from=a`. Then `bitwise_diff(load_state(ref), load_state(b))`:

```
[MEASURED] tests/gate0/test_g02_resume.py::test_resume_is_bitwise_on_all_four_components
           diff list is EMPTY; step == 5
```

| component | how compared | bitwise equal |
|---|---|---|
| model parameters (15 tensors) | `torch.equal` per tensor | **yes** |
| optimizer `exp_avg` (15) | `torch.equal` per tensor | **yes** |
| optimizer `exp_avg_sq` (15) | `torch.equal` per tensor | **yes** |
| optimizer `step` (15) | `torch.equal`, `tensor(5.)` | **yes** |
| optimizer `param_groups` | scalar equality | **yes** |
| `torch_rng_state` (5,056 B) | `torch.equal` | **yes** |
| `data_gen_state` (5,056 B) | `torch.equal` | **yes** |

**CUDA confirmation.** `[MEASURED]` The same k=2/m=3 comparison on
`device="cuda"`, RTX 4060 Laptop GPU, 6.94 GiB free of 7.996 GiB at the time of
the run: **diff count 0**. Run as a one-off script and deliberately **not** added
to the suite, so it never contends with the agent who owns device timing. This
is the tiny shape, not a shape that binds the card; it establishes that the
resume path is bitwise on CUDA at this size and nothing about larger shapes.

**None of this required a code change.** Resume was already bitwise on all four
components; what was missing was the measurement. The G0.2 code is unchanged
except for the in-place guard in §7, which is a G0.3 concern.

---

## 5. The planted negatives

Each breaks exactly one restore, then re-runs the resume and reports which
components the comparator flags. `monkeypatch.undo()` runs **before** the states
are read back — see the harness note below.

| # | negative | injection | components that differed | fired |
|---|---|---|---|---|
| **a** | optimizer `load_state_dict` no-op'd (the one the contract names) | `AdamW.load_state_dict = lambda self, sd: None` | `model`, `optimizer` | **YES** |
| **b** | torch RNG state not restored | `torch.set_rng_state = lambda s: None`, after `torch.manual_seed(987)` | `torch_rng_state` **only** | **YES, on its own component only** |
| **c** | data-generator state not restored | `torch.load` returns a `data_gen_state` of a fresh `manual_seed(seed+1)` | `model`, `optimizer`, `data_gen_state` | **YES** |

**Negative (b) is the finding.** Breaking the RNG restore does **not** change a
single weight. It is caught only because `torch_rng_state` is itself one of the
four compared components. The mechanism, measured:

- `[MEASURED]` `CEQForCausalLM.from_pretrained` does **not** consume the global
  torch RNG (transformers initialises on meta and loads).
- `[MEASURED]` the training loop consumes the global torch RNG **zero** times
  per step: batches are drawn from the explicit `gen`, and the model has no
  dropout and no other stochastic op (`grep dropout|rand|bernoulli|randn` over
  `ceq/hf/modeling_ceq.py` and `ceq/attention.py` returns nothing on the
  forward path).

So on this training path `torch_rng_state` is **inert**: it is correctly saved
and correctly restored, and nothing downstream reads it. That is a statement
about the loop's determinism, not a defect — but it means a weights-only resume
check would call this restore's removal a pass. It also sets the trigger
condition: **the moment anyone adds dropout, stochastic depth, or any
`torch.rand*` to the forward path, negative (b) starts changing weights**, and
at that same moment the missing CUDA RNG state (§9) becomes a real bug rather
than a documented gap.

Negative (c) had to be injected one level earlier than planned:
`torch.Generator` is an immutable C type and `set_state` cannot be replaced
(`TypeError: cannot set 'set_state' attribute of immutable type
'torch._C.Generator'`), so the injection substitutes the checkpointed state with
the state `gen` would have held had the restore never run.

**A harness bug found and fixed in the negatives themselves**, recorded because
it is the exact shape of a vacuous control: negative (c)'s injection patches
`torch.load`, which is also how `load_state` reads a checkpoint off disk.
Left live during the comparison it rewrote **both** sides identically, and the
negative reported `['model', 'optimizer']` — i.e. reported itself as *not*
firing on `data_gen_state`. An injection still live during the measurement is
measuring the injection. `_ref_and_broken` now calls `monkeypatch.undo()` before
reading the states back.

**A second self-inflicted defect, caught by the repo's own guard and recorded
here rather than quietly fixed.** Both test files first reached the shared
comparators as `from conftest import bitwise_diff, load_state`, copying
`tests/chase/conftest.py`'s `requires_triton` form. That form is what
`tests/loop/test_conftest_import_is_order_dependent.py` exists to forbid: 24 of
26 test directories carry a `conftest.py` and only 2 carry an `__init__.py`, so
the bare name `conftest` binds to whichever directory collection order reached
first. The chase file cited as precedent is itself one of that guard's ten
standing failures — precedent for the bug, not for the fix. Adding the two files
grew `tests/loop`'s failure set by exactly two node IDs. The comparators now live
in `tests/gate0/helpers.py`, imported as `from tests.gate0.helpers import ...`,
with `tests/gate0/conftest.py` re-exporting the two names because
`tests/gate0/test_g08_autopilot.py` cites that path. Measured after the move:

```
$ python -m pytest tests/loop -q -p no:randomly
15 failed, 525 passed, 3 warnings in 201.15s
```

15 is the standing baseline and no `tests/gate0` node ID appears in it. The
general lesson, which is the mechanism and not the instance: **an existing file
doing something is not evidence that the thing is allowed, when the repo carries
a test that names that file as an offender.**

---

## 6. The dataloader claim, measured rather than argued

`test_resumed_run_draws_the_same_batches_the_uninterrupted_run_drew` wraps
`ByteBatches.batch` and records the actual input tensor of every step.

```
[MEASURED] uninterrupted run drew 5 batches; resumed run drew 3
           ref[2] == b[0], ref[3] == b[1], ref[4] == b[2]   (torch.equal, all)
           ref[0] != ref[1]                                  (so the loop is not free)
```

The docstring's claim holds as stated **for this dataloader**: `ByteBatches`
draws by index from `gen` and holds no cursor of its own, so the generator state
is the whole position. The claim is a property of `ByteBatches`, not of
`train()` — a dataloader with shuffled epochs, prefetch, or worker processes
would carry position outside `gen` and this equality would break. `train()`
constructs `ByteBatches` directly, so nothing can substitute one today.

---

## 7. G0.3 — atomic write and kill-mid-write

### 7.1 RED first

`tests/gate0/test_g03_persist.py`, before any change to `ceq/hf/train.py`:

```
$ python -m pytest tests/gate0/test_g03_persist.py -q --no-header --tb=line
FFFF.F.                                                                  [100%]
E   AssertionError: trainer_state.pt was not written through os.replace; seen renames: []
E   AttributeError: module 'ceq.hf.train' has no attribute '_atomic_torch_save'
E   Failed: DID NOT RAISE <class 'OSError'>
E   AssertionError: a truncated trainer_state.pt was left in
    ...\test_kill_mid_write_leaves_no_0\c1\trainer_state.pt: 57636 bytes,
    and torch.load on it raises
E   safetensors_rust.SafetensorError: Error while serializing: I/O error: The
    requested operation cannot be performed on a file with a user-mapped
    section open. (os error 1224)
5 failed, 2 passed, 2 warnings in 47.49s
```

And the combined RED across both files: `5 failed, 12 passed` — **every G0.2
test was green before any code change**, which is the honest form of §4's
result: G0.2 needed measurement, G0.3 needed a fix.

### 7.2 The defect

`torch.save(obj, path)` **truncates** `path` and then streams into it. The RED
run above caught the consequence at exactly the size it happens: a
`trainer_state.pt` of **57,636 bytes** — half of the 115,272-byte state — left
in place, existing, newest file in its directory, and raising on `torch.load`.
Present-but-unloadable is strictly worse than absent, because absent is a loud
`FileNotFoundError` while present is a directory that looks resumable.

### 7.3 The fix

`ceq/hf/train.py::_atomic_torch_save(obj, path)` — sibling temp file, `flush`,
`os.fsync`, `os.replace`. Sibling and not the system temp directory because
`os.replace` is atomic only within one filesystem; the test asserts
`dirname(src) == dirname(dst)` rather than trusting the comment.

The write is also **ordered**: `save_pretrained` runs first, `trainer_state.pt`
last and atomically. A directory holding a loadable `trainer_state.pt`
therefore has complete model shards by construction. **No DONE marker was added
and none is needed** — the certificate already exists.

### 7.4 Kill-mid-write evidence

| kill point | injection | last good checkpoint | resumes bitwise |
|---|---|---|---|
| after temp write, before rename | `os.replace` raises `OSError` for `trainer_state.pt` | `_dir_fingerprint` (sha256 of every file) **unchanged**; no `trainer_state.pt` in the half-committed directory | **yes**, empty diff vs the uninterrupted 5-step run |
| part way through the bytes | `torch.save` writes half the serialization, then raises | fingerprint **unchanged**; no `trainer_state.pt` in the half-committed directory | **yes**, empty diff |
| inside `save_pretrained` | truncated `model.safetensors` written, then raises | fingerprint **unchanged**; `model.safetensors` present, `trainer_state.pt` **absent** — the ordering invariant | **yes**, empty diff |

"Resumes bitwise" is the same `bitwise_diff(load_state(ref), load_state(after))`
over all four components as §4, not a loss comparison. That is the round's KILL
clause and it is discharged on the crash path, not only the clean path.

### 7.5 One data-loss path closed

`train(out_dir=X, resume_from=X)` now raises `ValueError`. `save_pretrained` is
not atomic and is not being made atomic; writing back into the directory being
resumed from overwrites the model shards of the very state being resumed. On
Windows it does not even get that far — safetensors raises `os error 1224`
against its own memory-mapped read, which is what the RED run above shows.
TRAINING.md 6.6 already says `out_dir` must be a NEW directory each chunk; this
makes that a refusal instead of a convention. **Other agents: an autopilot that
retries a chunk must pick a new `out_dir`, not reuse the one it resumed from.**

### 7.6 Periodic checkpoints -- the failure mode atomicity alone does not survive

The write was atomic but it happened **once, at the end of a chunk**, and a
Kaggle chunk runs up to 11 hours against a 30 GPU-h weekly quota
(`TRAINING.md:142`, `[INHERITED]`). A kernel death at hour 10 costs ten
GPU-hours; three of them is the week. The autopilot's Tier-1 rule
*"session <20 min & no ckpt in 30 -> force ckpt"* also presumes periodic
checkpoints exist and can be observed. Raised by the notebook and autopilot
agents, and correct: **the round is built around a failure mode an
end-of-chunk-only save does not address.**

`save_every=N` writes a complete checkpoint every N steps through the same
`_atomic_torch_save` path, into **two alternating slots**, `{out_dir}.ckpt-a`
and `{out_dir}.ckpt-b`.

**Two slots and not one**, because `save_pretrained` is not atomic: a periodic
save overwriting a single slot in place would destroy the only mid-chunk
checkpoint at exactly the moment it is being replaced -- the window a session
cap is most likely to land in on a long chunk. **Two and not N**, because N
needs a reaper and a reaper needs to decide what to delete; two is bounded by
construction, with no globbing and no deletion of anything `train()` did not
itself just write. Both slots are removed once `out_dir` loads, verified with
`_ckpt_step` rather than assumed.

8 tests, all RED before the change (`TypeError: train() got an unexpected
keyword argument 'save_every'`, `AttributeError: module 'ceq.hf.train' has no
attribute '_save_checkpoint'` / `'latest_checkpoint'`):

| test | what it holds |
|---|---|
| `test_resume_from_a_mid_chunk_periodic_checkpoint_is_bitwise` | **the case the Kaggle run actually exercises.** Kill after the step-2 periodic save; `latest_checkpoint` returns `.ckpt-a` at step 2; resuming 3 more steps is bitwise-equal to the uninterrupted 5-step run on all four components. Every other resume test in this tree resumes from an *end-of-chunk* save; a session cap does not land on a chunk boundary. |
| `test_periodic_saving_does_not_change_the_end_state` | `save_every=2` and `save_every=0` end bitwise-identical. A checkpoint is an observation and must not perturb what it observes. |
| `test_slots_alternate_so_a_kill_during_one_leaves_the_other` | kill during the **second** periodic save: `.ckpt-b` is unloadable, `.ckpt-a` still stands at step 2, `latest_checkpoint` picks it, and resuming from it is bitwise. |
| `test_latest_checkpoint_prefers_the_highest_step_and_skips_the_corrupt` | highest step wins; a corrupted winner is skipped rather than raising or being picked anyway; an empty root returns `None`. |
| `test_slots_are_removed_once_the_chunk_directory_is_complete` | the disk bound in §8. |
| `test_refuses_a_resume_from_its_own_periodic_slot` | `ValueError`. `{out_dir}.ckpt-a` is a name the chunk will overwrite at step `save_every`, so resuming from it into the same chunk destroys the source while the run is healthy. |
| `test_save_every_writes_periodic_checkpoints` | the end-of-chunk directory still lands at the full step count. |
| `test_save_every_is_in_the_signature_for_the_notebook_probe` | keyword-only, default `0`. The notebook detects it by `inspect.signature`. |

### 7.7 A latent defect in the probe, found by the first mid-loop save

The first periodic checkpoint died with `RecursionError: maximum recursion depth
exceeded` inside `Module.state_dict()`. Cause, in code that predates this work:

```python
layer.self_attn.qkv._ceq_owner = layer.self_attn      # before
```

Assigning an `nn.Module` to an attribute of another `nn.Module` **registers it
as a submodule**. `qkv` therefore contained `self_attn`, which contained `qkv` --
a cycle in the module tree. It had never fired because the only
`save_pretrained` ran *after* the post-loop `del layer.self_attn.qkv._ceq_owner`,
so nothing ever walked the tree while the probe was attached. The first
mid-loop save walks it immediately.

Fixed at the root rather than worked around: `_attach_row_l1_probe` now binds
the owning attention in a **closure** (`make_hook(attn)`), which carries the same
reference with no registration. The model tree is unchanged while the probe is
on, and the post-loop `del` loop is gone with it -- a smaller function than
before. Grep confirmed `_ceq_owner` had exactly three references, all in this
file. The alternative -- deleting and restoring the attribute around every save
-- would have left the cycle in place for the next caller to trip over.

**A repo mistake worth cataloguing by mechanism**, not by instance: *state
stored on an `nn.Module` by attribute assignment silently joins the module tree,
and a defect there stays invisible until something serializes the model at a
moment nobody had serialized it before.*

---

## 8. Rotation sized to `/kaggle/working`

`[MEASURED]` One **real** checkpoint written by `train(steps=1, **train.DEFAULTS)`
on CPU — the notebook training shape, hidden 512 / L8 / H8 / seq 512 / batch 8 /
vocab 256, **25,728,000 real parameters** (`TRAINING.md:65-73`, `[INHERITED]`
cited by file; `ceq/sizing.py::params` gives 25,707,520 analytic, the 20,480
difference being the MLP bias the formula omits).
Reproduce with:

```
python -m pytest "tests/gate0/test_g03_persist.py::test_rotation_arithmetic_at_the_real_training_shape" -q -s
```

```
[MEASURED] one checkpoint at train.DEFAULTS
  config.json                            676 B
  configuration_ceq.py                14,635 B
  generation_config.json                 160 B
  model.safetensors              102,920,784 B
  modeling_ceq.py                     38,307 B
  run_record.json                        278 B
  trainer_state.pt               205,902,768 B
  TOTAL                          308,877,608 B = 0.2877 GiB
```

Re-measured after the periodic-checkpoint work. The total moved 10,499 B from
308,867,109 B because two other agents grew `configuration_ceq.py` and
`modeling_ceq.py` in the meantime; the two large files are unmoved and the
checkpoint count is 64 either way.

The two large files check out against the parameter count, which is why they are
not just quoted: `25,728,000 × 4 = 102,912,000` fp32 bytes plus an 8,784-byte
safetensors header = `model.safetensors`; `2 × 102,912,000 = 205,824,000` for
AdamW's two moments plus 15 `step` scalars, two 5,056-byte RNG buffers and
pickle framing = `trainer_state.pt`. **A checkpoint is 12 bytes per parameter**,
3× the model.

Arithmetic, inputs named:

| input | value | tag |
|---|---|---|
| `/kaggle/working` budget | 20 GB = 20,000,000,000 B | `[ASSUMED]` — `TRAINING.md:142` states "20 GB auto-saved `/kaggle/working`"; read as decimal GB, as storage quotas are stated. At 20 GiB the counts below are 7.4 % larger, so decimal is the conservative reading. |
| one checkpoint | 308,877,608 B | `[MEASURED]` |
| in-flight `.tmp` | 205,902,768 B | `[MEASURED]` — `_atomic_torch_save`'s sibling, one per directory, live only during the write |

```
[MEASURED] checkpoints that fit = floor(20,000,000,000 / 308,877,608) = 64
[MEASURED] last good + running chunk               =   617,755,216 B = 0.5753 GiB = 3.0888 %
[MEASURED] + the running chunk's 2 periodic slots  = 1,235,510,432 B = 1.1507 GiB = 6.1776 %
[MEASURED] + one in-flight .tmp (peak)             = 1,441,413,200 B = 1.3424 GiB = 7.2071 %
```

**The rotation policy**, which is TRAINING.md 6.6's new-directory-per-chunk made
explicit:

1. Chunk *n+1* writes to a **new** `out_dir`, resuming from chunk *n*. Never
   into chunk *n* — `train()` now refuses that (§7.5).
2. Chunk *n*'s directory is deleted **only after** chunk *n+1*'s directory holds
   a `trainer_state.pt` that `torch.load` opens. That single condition certifies
   the whole directory (§7.3), so the check is one `torch.load` in a `try`.
3. Within the running chunk, `save_every=N` keeps **two** periodic slots,
   alternating. `train()` deletes both itself once `out_dir` is verifiably
   complete, so they never outlive their chunk.
4. Steady-state occupancy is therefore **four** directories — two chunk
   directories plus the running chunk's two slots — at **6.18 %** of the budget,
   **7.21 %** at the instant of a write. 64 is the ceiling if nothing is ever
   deleted; at the 20,000-steps-per-chunk setting of TRAINING.md 6.5 that
   ceiling is not reachable inside the 30 GPU-h weekly quota anyway.
5. A crashed chunk leaves at most one stale `trainer_state.pt.tmp`
   (205,902,768 B) per directory; deleting that directory reclaims it.

**No reaper was written**, and none is needed for the slots — `train()` removes
the two names it wrote itself, after verifying `out_dir` loads. Deleting the
*previous chunk's* directory (step 2) is left to the caller that owns the chunk
loop: it is one `shutil.rmtree` guarded by one `latest_checkpoint` call, and a
rotation manager that deletes the user's directories on its own judgement is the
kind of thing that gets someone paged at 3am.

---

## 9. Limits

- **Bitwise is `torch.equal`, per the contract, not a byte comparison.**
  `torch.equal` calls `+0.0` and `-0.0` equal although their bits differ, and
  calls NaN unequal to itself although the bits agree. Neither is reachable from
  AdamW moments or uint8 RNG buffers, and neither appeared; a byte-level
  comparison would be strictly stronger and was not done.
- **`torch.cuda.get_rng_state()` is NOT checkpointed.** `trainer_state.pt` holds
  the CPU RNG state only. Harmless today for the reason in §5 — the forward path
  has no CUDA-side stochastic op — and a real bug the day dropout is added. Not
  fixed here: adding a fifth key changes the checkpoint format that
  `TRAINING.md:364` documents as exactly `['data_gen_state', 'optimizer',
  'step', 'torch_rng_state']`, and that file is outside this item's ownership.
- **Shape.** Bitwise equality is established at 11,488 parameters on CPU and on
  CUDA, and at 25,728,000 parameters only for the checkpoint's *byte size*, not
  for its bitwise resume. A 20,000-step CUDA chunk at the training shape has not
  been resumed and compared, and would take hours on this box.
- **`save_pretrained` is not atomic.** A kill inside it leaves a truncated
  `model.safetensors`. That is contained, not fixed: the directory is
  identifiable as bad because `trainer_state.pt` never appears in it, the
  previous directory is untouched, and for periodic saves the *other* slot is
  always a complete checkpoint. Making it atomic would mean writing shards to
  temp names and renaming, which is a construction this round did not ask for.
- **No directory fsync after the rename.** There is no portable way to fsync a
  directory on Windows, where this was developed. A host power cut could lose
  the rename itself; the previous checkpoint directory is then the last good
  one, which the policy already assumes. Flagged with a `ponytail:` comment at
  the call site.
- **`tests/gate0/helpers.py` is shared.** Four other agents are editing this
  tree and `tests/gate0/test_g08_autopilot.py` already documents a dependency on
  `load_state` and `bitwise_diff`. Those two names must stay, and
  `tests/gate0/conftest.py` re-exports them so that citation keeps resolving.
- **L-TIME.** Every wall-clock figure here was measured in this session:
  chase resume 11.05 s → 13.30 s, gate0 RED 47.49 s, gate0 GREEN 68.28 s,
  rotation measurement 33.23 s. Step timings are **NOT MEASURED** in this work
  and none are quoted.

---

## 10. Interface other agents can call

`ceq/hf/train.py::train()` is unchanged in signature. It was already
programmatically callable and remains the resume entry point:

```python
train(*, out_dir, steps, batch, seq, hidden_size, n_layers, n_heads,
      device="cuda", vocab_size=256, data_path=None, max_bytes=64*1024*1024,
      lr=3e-4, seed=0, grad_checkpoint=False, clip=1.0, probe_every=1,
      log_every=50, gpu=None, resume_from=None, save_every=0) -> dict

latest_checkpoint(out_dir) -> str | None
```

`save_every` is the only signature change and it is **keyword-only with default
`0`**, so every existing call site is unaffected. The notebook's
`inspect.signature` probe is pinned by
`test_save_every_is_in_the_signature_for_the_notebook_probe`.

Resume contract:

- `resume_from` is an **earlier call's `out_dir`, or either of its periodic
  slots**. `None` starts fresh. The autopilot passes `event.last_good_ckpt`;
  produce that with `latest_checkpoint(prev_out_dir)`.
- `steps` is steps **ADDED**, not a new total. `seed` is ignored on resume.
  Resuming from a mid-chunk checkpoint at step 8,000 with `steps=20000` ends at
  28,000, not 20,000.
- `out_dir` **must not equal** `resume_from`, **nor may either of
  `{out_dir}.ckpt-a` / `.ckpt-b`** — `ValueError` since this work. Use a new
  `out_dir` per chunk.
- `save_every=N` writes a checkpoint every N steps into the two alternating
  slots, through the atomic path. `0` (default) is end-of-chunk only. The slots
  are deleted by `train()` once `out_dir` is verified complete.
- Returns a record containing `start_step` (== the step the resume source stood
  at), `steps`, `losses`, `grad_norms`, `row_l1_min`, `peak_bytes`, `n_params`.
- A directory is resumable **iff** `{dir}/trainer_state.pt` exists and
  `torch.load(..., weights_only=True)` opens it. That is the whole check; the
  model shards are complete by construction when it passes.
  `latest_checkpoint(out_dir)` applies it to the chunk directory and both slots
  and returns the highest-step winner, or `None`.
- Module-level and public: **`latest_checkpoint(out_dir)`**.
- Module-level and private, listed because the tests drive them:
  `_atomic_torch_save(obj, path)`, `_save_checkpoint(model, opt, gen, step,
  out_dir)`, `_ckpt_step(out_dir)`.
