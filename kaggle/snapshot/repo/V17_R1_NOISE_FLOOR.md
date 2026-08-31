# V17-K RULING 1 (+ RULING 2a) — THE TRAINING NOISE FLOOR HARNESS

**LEAD CAVEAT, stated once so it does not have to be repeated per section.**
The measurement itself did not happen here. `V17_GPU_QUEUE.md`: *"it bounds
training claims, and training runs on Kaggle. Measuring it locally would bound
the wrong device. Not a deciding number."* This node built the harness, proved
it on CPU at a tiny shape, and hands it to the notebook. Every number in this
file is either `[MEASURED]` on **this box, this run, CPU, throwaway shape** —
never cited as the floor — or `NOT MEASURED — KAGGLE`. **No `kaggle` CLI call,
no upload, no launch, no touch of `~/.kaggle`; no writing git command.**

## Provenance

| | `git rev-parse HEAD` | `git status --porcelain` |
|---|---|---|
| **start** | `ab5b48547884e04258276e6e808d5a71ea65f917` | `M MODEL_CARD.md`, `M ceq/arm_smprime.py`, `M ceq/hf/configuration_ceq.py`, `M ceq/hf/modeling_ceq.py`, `M ceq/hf/train.py`, `M house-events.jsonl`, `M scale/identity_manifest.py`, `M scripts/v15_r1.py`; untracked `COSTS.md`, `V17K_RULINGS.md`, `V17_ARM_WIRING.md`, `V17_G01_IDENTITY.md`, `V17_G02_G03_CHECKPOINT.md`, `V17_G04_ENV.md`, `V17_G05_DATA.md`, `V17_G06_G07_CERT_COST.md`, `V17_G08_G09_AUTOPILOT.md`, `V17_GPU_QUEUE.md`, `V17_LABEL_CELL_REPAIR.md`, `V17_NOTEBOOK.md`, `V17_R1_R3_R7_EDITS.md`, `ceq/autopilot.py`, `ceq/kdata.py`, `kaggle/`, `requirements-kaggle.txt`, `results/k_cert_local.json`, `results/k_data_manifest.json`, `scripts/k_cert.py`, `scripts/k_cost.py`, `tests/gate0/` |
| **end** | `ab5b48547884e04258276e6e808d5a71ea65f917` (**unchanged**) | same, plus `?? scripts/k_noise_floor.py` and this file, plus other nodes' concurrent landings (`V17_R4_RETAKE_PRICE.md`, `V17_R5_INSTRUMENT.md`, `V17_R6_ENVELOPE.md`, `results/r4_price_probe.json`) — **none touched by this node** |

**Files this node wrote — the only three it was allowed to touch:**
`scripts/k_noise_floor.py` (new), `tests/gate0/test_g15_noise_floor.py` (new),
this file. `ceq/hf/train.py` was **read, not edited**; the `M` against it above
predates this session (another node's work).

---

## 1. WHAT THE HARNESS MEASURES, AND UNDER WHICH REGIME

`V17K_RULINGS.md` RULING 1, verbatim: two training chunks from an **identical
seed** and identical configuration, run under CUDA with
`torch.use_deterministic_algorithms(True, warn_only=True)` — strict mode is
not executable in the backward, because autograd differentiates `cumprod`
with `cumsum`, which has no deterministic CUDA kernel (`COSTS.md` §1.6
measures this directly: `RuntimeError: cumsum_cuda_kernel does not have a
deterministic implementation`, raised through the backward of `cumprod`). The
`|Δ|` between the two chunks' **final losses** is the noise floor — the amount
by which two runs that *should* be identical actually differ. Every training
claim in the round is then held to that floor, never to bitwise equality
(L-TOL, as amended: bitwise for replay + forward; the measured floor for
training).

**RULING 2a, filed mid-task and folded in here** (`V17K_RULINGS.md`, new
section after §A4): the loss-space floor is **one instance of a general rule**,
now law —

> *A quantity's noise floor is measured in that quantity's units from the
> identical-seed pair.*

The round's immediate use of the general rule is `δ_β,i = |β⁽ᵃ⁾ᵢ − β⁽ᵇ⁾ᵢ|`,
`smprime`'s `beta` (one scalar `nn.Parameter` per layer —
`ceq/hf/modeling_ceq.py::beta_column`'s own docstring), which is RULING 2a's
pinning criterion `|β_i,final − 1| ≤ 5·δ_β,i` — β's own noise floor in β's own
units, from the same pair, no new run. The harness below produces both from
one call.

`scripts/k_noise_floor.py` sets the regime with
`torch.use_deterministic_algorithms(True, warn_only=True)` immediately before
running either chunk, restores whatever was in force before the call in a
`finally`, and **reads the regime back** (`torch.are_deterministic_algorithms_
enabled()`, `torch.is_deterministic_algorithms_warn_only_enabled()`) rather
than assuming the request took — the JSON record's `regime` field is what was
**actually** in force, not what was asked for. `CUBLAS_WORKSPACE_CONFIG` is set
before any `torch` import, same reasoning and same value (`":4096:8"`) as
`scripts/k_cert.py`.

---

## 2. THE ONE CALL

```python
from scripts.k_noise_floor import measure_noise_floor

record = measure_noise_floor(
    out_dir_a="/kaggle/working/nf_chunk_a",
    out_dir_b="/kaggle/working/nf_chunk_b",
    seed=0,
    steps=204,                 # or whatever Q3's chunk-table entry names
    batch=8, seq=512, hidden_size=512, n_layers=8, n_heads=8,
    vocab_size=256,
    device="cuda",
    operator="smprime",        # forwarded to train() like any other override
    delta_param_names="self_attn.beta",   # RULING 2a: attach δ_β,i per layer
)
```

That is the whole interface. `measure_noise_floor` runs **both** chunks
through `ceq.hf.train.train()` itself (`ceq/hf/train.py` is not edited —
imported and called exactly as it stands), applies and restores the
determinism regime around them, checks the pair, and returns one dict —
`json.dumps(record, indent=2)` is the file `COSTS.md` §4 ingests. Nothing else
for the notebook to assemble; no second call, no manual `|Δ|`, no manual
regime bookkeeping.

`out_dir_a`/`out_dir_b` are two **fresh, distinct** directories — the harness
refuses if they resolve to the same path (§3.2). No `resume_from` is ever
passed; both chunks train from scratch under the same seed, because RULING 1's
class B1 (resume/replay, bitwise) is a different claim, already MET elsewhere
(`V17_G02_G03_CHECKPOINT.md` §1), and is not what this measures.

`latest_checkpoint(out_dir)` was **not needed**: this measurement is two
independent from-scratch chunks under one seed, not a resume chain, so
`ceq.hf.train.py`'s resume machinery has no role here. Stated for the record
per the task's ask to name what was and was not used from that interface —
**no addition to `train.py` was required.**

A CLI wrapper exists for parity with `scripts/k_cert.py` / `scripts/k_cost.py`
and local smoke-testing (`python scripts/k_noise_floor.py --device cpu ...`);
the notebook is expected to call the function directly.

---

## 3. THE THREE REFUSALS

### 3.1 `assert_identical_pair` — RULING 1's "identical seed" clause

> "If the two chunks differ in seed, shape, step count, data or regime, it
> raises rather than returning a number."

`_MUST_MATCH = (seed, hidden_size, n_layers, n_heads, seq, batch, vocab_size,
steps, data_path, operator, device, deterministic_algorithms, warn_only)` —
every field the ruling names, and no more. It is checked **after** both chunks
run, against each chunk's **own recorded provenance** (`operator`/`device`/
`steps` read back off `run_record.json`, not the caller's input arguments) —
so a silent divergence (operator resolving differently, device falling back,
an interrupted short run) is still caught even though the public call only
accepts one seed/shape for both chunks by construction. It is a pure,
standalone function, unit-tested against a synthetic mismatch **for every one
of the 13 fields** (`tests/gate0/test_g15_noise_floor.py::
test_assert_identical_pair_raises_on_every_required_field`, parametrized) —
each proven to fire before the matching case is trusted to pass.

### 3.2 `assert_not_self_comparison` — telling a real `|Δ|=0` from a bug

**This is the load-bearing check the task asked for.** Two chunks that return
`|Δ| = 0.0` could mean determinism held, or could mean the harness compared a
chunk to itself. Two structural failure modes, both checked **on disk**, the
same reason `tests/gate0/helpers.py::load_state`'s own docstring gives for
reading a checkpoint back off disk rather than trusting the live objects a run
left in memory:

1. **SAME DIRECTORY.** `out_dir_a == out_dir_b` → the second `train()` call
   overwrote the first chunk's own record; there is only ever one run on disk,
   and any `|Δ|` computed from it is 0 **by construction**, not measurement.
   Checked with `os.path.abspath`, before either chunk trains.
2. **STALE REUSE.** One of the two directories already held a
   `run_record.json` from an earlier, unrelated invocation, and this call
   never actually wrote a fresh one into it — a caller pointing the harness at
   a leftover directory, or a bug that silently skips the second `train()`
   call. `t_call_start`, captured before either chunk runs, is the line: a
   genuine run of *this* call leaves both `run_record.json` mtimes at or after
   it. Older than that ⇒ raise.

The final losses used for `|Δ|` are then read **from those same on-disk
records, keyed only by `out_dir`** (`_read_run_record`), never from the
in-memory `record_a`/`record_b` objects `train()` happened to return — so a
`final_b` that quietly reused `record_a` cannot even compile, let alone run.

Both failure modes have their own must-fire test, paired with a
non-degeneracy pass so the check is shown to say **both** yes and no:
`test_assert_not_self_comparison_raises_on_same_directory`,
`test_assert_not_self_comparison_raises_on_a_stale_directory` (a real
directory, mtime forced before `t_call_start` by writing it first and sleeping
across the boundary), and
`test_assert_not_self_comparison_passes_on_two_fresh_distinct_dirs`. The
integration-level version — the one public call refusing the same shape of
bug directly, not only the helper in isolation —
is `test_measure_noise_floor_refuses_out_dir_a_equal_out_dir_b`.

**Why a genuine `|Δ|=0` on CPU is trustworthy despite reading `0.0` on the
nose:** because the self-comparison path *raises before a number is ever
produced*. A returned `0.0` can therefore only have come from two distinct,
freshly-written directories — the alternative path does not return, it
throws. `test_two_independent_cpu_chunks_at_a_fixed_seed_reproduce_bitwise`
asserts the harness's own CPU determinism (`trajectory_bitwise_identical is
True`, not merely the final loss); its non-degeneracy pair,
`test_non_degeneracy_losses_actually_move_across_steps`, confirms the losses
being compared are not sitting at a constant for an unrelated reason (`len(set
(losses)) > 1`).

### 3.3 `assert_matching_parameters` — RULING 2a's extension

> "The two runs must have the same parameter names and shapes, or the keyed
> comparison is meaningless — mismatched keys must raise, never silently
> intersect."

`assert_matching_parameters(params_a, params_b)` requires `set(params_a) ==
set(params_b)` (raising both directions' extra names on mismatch) and matched
`.shape` per name. **The must-fire test that a naive `set(a) & set(b)`
intersection would defeat:**
`test_assert_matching_parameters_never_silently_intersects` constructs two
dicts that **share** one key and each carry one name the other lacks — a
buggy intersection-based implementation passes the two single-sided tests
next to it while quietly dropping both extra names; this one does not let it.

`per_parameter_deltas(out_dir_a, out_dir_b, names=None)` is the general
primitive this guards: `|Δ|`, elementwise, keyed by parameter name, read from
the **same** identical-seed pair `measure_noise_floor` already trained — no
new run, and not hardcoded to `beta`. `names` is `None` (every parameter), a
string (substring filter — `"self_attn.beta"` picks up every layer without
the caller enumerating indices), or an explicit list (each name required to
exist, or it raises rather than being silently dropped —
`test_per_parameter_deltas_raises_on_an_unknown_exact_name`). It repeats
refusal 3.2's same-directory check on its own cheap half, so it stays safe to
call standalone on any two checkpoints later, for any quantity, without a new
harness.

**Per-parameter zero told apart from a self-comparison, the same way as
§3.2:** `test_per_parameter_deltas_refuses_out_dir_a_equal_out_dir_b` (raises,
same-directory, before any checkpoint loads — the path need not even exist)
paired with `test_per_parameter_zero_on_cpu_is_told_apart_from_self_comparison`
(two real, distinct, freshly-trained CPU checkpoints, every layer's `δ_β`
reads `0.0`, and is trustworthy for the identical reason: the self-compared
path is the one that never returns a number).

---

## 4. THE CPU PROOF RUN — HARNESS TEST, NOT THE FLOOR

`measurement_validity` is computed by `_floor_label(device)`: `"FLOOR"` only
when `device == "cuda"`; every other device reads
`"HARNESS_TEST_ONLY -- NOT THE FLOOR (device=... != 'cuda', V17_GPU_QUEUE.md:
the measurement runs on Kaggle)"`. Unit-tested both directions
(`test_cpu_proof_is_never_labelled_a_floor`) so the label is shown to depend
on the device rather than being asserted only in the negative on a CPU-only
run.

**Actual CLI smoke test, this box, CPU, throwaway shape** (`hidden_size=16,
n_layers=1, n_heads=2, seq=8, batch=2, steps=3`, 47-line synthetic corpus) —
full JSON in `scripts/k_noise_floor.py`'s own doctest-free smoke path, key
fields reproduced here `[MEASURED, CPU, HARNESS TEST — NOT THE FLOOR]`:

```json
{
  "measurement_validity": "HARNESS_TEST_ONLY -- NOT THE FLOOR (device='cpu' != 'cuda', V17_GPU_QUEUE.md: the measurement runs on Kaggle)",
  "regime": {"deterministic_algorithms": true, "warn_only": true, "cublas_workspace_config": ":4096:8"},
  "step_count": 3,
  "chunk_a": {"final_loss": 4.128396511077881, "n_losses": 3},
  "chunk_b": {"final_loss": 4.128396511077881, "n_losses": 3},
  "abs_delta_final_loss": 0.0,
  "trajectory_bitwise_identical": true,
  "identical_pair_check": "PASSED"
}
```

A second CLI run at `operator="smprime"`, `n_layers=2`, `--delta-param-filter
self_attn.beta`, same tiny CPU shape, adds RULING 2a's field
`[MEASURED, CPU, HARNESS TEST]`:

```json
"per_parameter_delta": {
  "model.layers.0.self_attn.beta": {"shape": [], "delta": 0.0, "max_abs_delta": 0.0, "mean_abs_delta": 0.0},
  "model.layers.1.self_attn.beta": {"shape": [], "delta": 0.0, "max_abs_delta": 0.0, "mean_abs_delta": 0.0}
}
```

**Both `0.0` readings are the harness proving itself on CPU, where real
bitwise reproducibility is expected (no CUDA backward hole exists to
measure) — neither is `FLOOR_TRAIN_ABS_DLOSS` or `FLOOR_BETA_DELTA`, and
`measurement_validity` says so on the record itself so nobody downstream can
cite either as RULING 1's or RULING 2a's number by mistake.** The floor is
`NOT MEASURED — KAGGLE`; it exists only when this same call runs with
`device="cuda"` on the certified box, on Kaggle.

---

## 5. THE JSON SCHEMA `COSTS.md` §4 INGESTS

`schema: "k_noise_floor/1"`.

```
{
  "schema": "k_noise_floor/1",
  "measurement": "R1_TRAINING_NOISE_FLOOR",
  "measurement_validity": "FLOOR" | "HARNESS_TEST_ONLY -- NOT THE FLOOR (...)",
  "seed": int,
  "config": {hidden_size, n_layers, n_heads, seq, batch, vocab_size, steps,
             data_path, operator, overrides: {...}},
  "regime": {deterministic_algorithms: bool, warn_only: bool,
             cublas_workspace_config: str|null},
  "box": { ... scripts.k_cert.box_record(device) -- device_name, torch,
           python, platform, cuda_available, tf32 flags, capability,
           total_vram_bytes, n_gpus },
  "step_count": int,
  "chunk_a": {out_dir, final_loss, n_losses, run_record_mtime},
  "chunk_b": {out_dir, final_loss, n_losses, run_record_mtime},
  "abs_delta_final_loss": float,          # <- FLOOR_TRAIN_ABS_DLOSS
  "trajectory_bitwise_identical": bool,
  "per_parameter_delta": null | {
      "<parameter name>": {"shape": [...], "delta": float | nested list,
                            "max_abs_delta": float, "mean_abs_delta": float}
  },                                        # <- FLOOR_BETA_DELTA, keyed by name
  "identical_pair_check": "PASSED",
  "self_comparison_check": {out_dir_a, out_dir_b, run_record_mtime_a,
                            run_record_mtime_b, t_call_start},
  "git": {head, porcelain},
  "elapsed_s": float
}
```

**`FLOOR_TRAIN_ABS_DLOSS`** (`V17K_RULINGS.md` §A1.2) = `abs_delta_final_loss`
on a `device="cuda"` run. **`FLOOR_CHUNK_SPEC`** (§A1.2) = `config` + `regime`
+ `seed` + `step_count`, all present on the same record — the chunk spec
travels *with* the number rather than being a constant applied elsewhere
(the exact V-22 class `V16_DEVICE_CERT.md` §3.2 warns against).

**The field the β node (`ceq/hf/modeling_ceq.py`) needs, stated so the two
schemas meet:** `record["per_parameter_delta"]["model.layers.{i}.self_attn.
beta"]["delta"]` is `δ_β,i` for layer `i` — a bare float (the parameter is a
0-d tensor, `shape: []`), not a list. Call with `delta_param_names=
"self_attn.beta"` to populate it for every layer in one shot; `max_abs_delta`
equals `delta` for a scalar parameter (kept as a separate field so a future
non-scalar quantity's summary has the same shape). If the β node's own
per-instance layout differs from `ceq/hf/modeling_ceq.py`'s current
`self_attn.beta` naming, `per_parameter_deltas`' `names=` filter is a plain
substring match — no change to this file is needed to point it at a renamed
parameter, only a different string at the call site.

---

## 6. TDD EVIDENCE

### RED — round 1, the harness itself (module did not exist)

```
=================================== ERRORS ====================================
____________ ERROR collecting tests/gate0/test_g15_noise_floor.py _____________
ImportError while importing test module '...\tests\gate0\test_g15_noise_floor.py'.
tests\gate0\test_g15_noise_floor.py:44: in <module>
    from scripts import k_noise_floor as knf
E   ImportError: cannot import name 'k_noise_floor' from 'scripts' (unknown location)
=========================== short test summary info ===========================
ERROR tests/gate0/test_g15_noise_floor.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.82s
```

### GREEN — round 1, 23/23

```
23 passed, 2 warnings in 7.48s
```

### RED — round 2, RULING 2a landed mid-task; tests written before the code

```
FAILED tests/gate0/test_g15_noise_floor.py::test_assert_matching_parameters_passes_on_matched_shapes
FAILED tests/gate0/test_g15_noise_floor.py::test_assert_matching_parameters_raises_on_a_name_only_in_a
FAILED tests/gate0/test_g15_noise_floor.py::test_assert_matching_parameters_raises_on_a_name_only_in_b
FAILED tests/gate0/test_g15_noise_floor.py::test_assert_matching_parameters_never_silently_intersects
FAILED tests/gate0/test_g15_noise_floor.py::test_assert_matching_parameters_raises_on_shape_mismatch
FAILED tests/gate0/test_g15_noise_floor.py::test_per_parameter_deltas_refuses_out_dir_a_equal_out_dir_b
FAILED tests/gate0/test_g15_noise_floor.py::test_per_parameter_deltas_end_to_end_with_substring_filter
FAILED tests/gate0/test_g15_noise_floor.py::test_per_parameter_deltas_raises_on_an_unknown_exact_name
FAILED tests/gate0/test_g15_noise_floor.py::test_measure_noise_floor_attaches_per_parameter_delta_only_when_requested
FAILED tests/gate0/test_g15_noise_floor.py::test_per_parameter_zero_on_cpu_is_told_apart_from_self_comparison
10 failed, 23 deselected, 2 warnings in 6.55s
```

(`KeyError: 'per_parameter_delta'` and `AttributeError: module 'scripts.
k_noise_floor' has no attribute 'assert_matching_parameters'` / `'per_
parameter_deltas'` — the functions and field did not exist yet.)

### GREEN — full suite, 33/33

```
tests/gate0/test_g15_noise_floor.py .................................  [100%]
33 passed, 2 warnings in 8.75s
```

**A real flake was caught and fixed during repeat verification, not left in.**
Re-running the suite surfaced
`test_assert_not_self_comparison_passes_on_two_fresh_distinct_dirs` failing
roughly 2 times in 5. Measured directly (a 10-trial script writing a file
immediately after capturing `time.time()`): on this filesystem a
`run_record.json`'s `os.path.getmtime()` can land a hair below the
`t_call_start` captured microseconds earlier — clock/filesystem-mtime
quantization, not real staleness. `assert_not_self_comparison`'s freshness
check now allows `_FRESHNESS_SLACK_S = 0.5` seconds of that quantization
(a genuinely stale, reused directory is older by seconds-to-weeks, so the
slack costs the staleness check nothing it exists to catch — the paired
must-fire test now sleeps `_FRESHNESS_SLACK_S + 0.3` s on each side of
`t_call_start` to stay clear of it). The test's own `>=` assertions on
`run_record_mtime_a/b` were tightened to match — a strict `>= t_call_start`
there was reintroducing the identical flake one layer up. Re-run 8x after
the fix, 8/8 green (`4 passed` each, the self-comparison subset); full
33-test suite re-run clean afterward.

**Non-degeneracy is paired into every PASS half**, per the task's own
requirement — not asserted separately at the end:

| PASS | its non-degeneracy pair |
|---|---|
| `assert_identical_pair` matches | fires on all 13 fields, parametrized |
| `assert_not_self_comparison` clears two fresh dirs | fires on same-dir, fires on stale-dir |
| `assert_matching_parameters` clears matched shapes | fires on extra-in-A, extra-in-B, **shared-key-with-one-extra-each** (defeats a naive intersection), shape mismatch |
| CPU chunks reproduce bitwise (`\|Δ\|=0`) | losses actually move across steps (not a constant) |
| per-parameter `δ_β=0` on two real dirs | same function raises on one dir passed twice |
| CPU proof labelled `HARNESS_TEST_ONLY` | `_floor_label("cuda") == "FLOOR"` |

Full suite alongside the rest of Gate-0: `pytest tests/gate0/ --collect-only`
reads **245 tests collected**, no collection errors, after both rounds — up
from 235 after round 1 alone (23 tests) and, before this file existed, from
whatever the other nodes' files contributed; the round-1→round-2 delta is
exactly this file's own 10 RULING-2a additions (235 + 10 = 245).

---

## 7. INTERFACE NEEDED FROM `ceq/hf/train.py`

**None.** `train(*, out_dir, steps, batch, seq, hidden_size, n_layers,
n_heads, device, vocab_size, data_path, seed, resume_from, save_every,
**overrides)` and its `run_record` (`losses`, `operator`, `steps`, `device`)
as it stands today were sufficient for both the loss-space floor and the
per-parameter extension — the latter reads checkpoints back with
`CEQForCausalLM.from_pretrained`, already how
`tests/gate0/helpers.py::load_state` and `ceq/hf/modeling_ceq.py::beta_column`
do it. `latest_checkpoint` was not needed (§2). No change requested.

---

## 8. GREEN / RED / BLOCKED

- **GREEN.** The harness (`scripts/k_noise_floor.py`), both refusals RULING 1
  required and the two RULING 2a added, and the CPU proof — all built, all
  tested RED-then-GREEN, 33/33, non-degenerate on every load-bearing check,
  confined to the three files this node owns, no forbidden file touched, no
  writing git command run, HEAD unchanged at `ab5b485`.
- **NOT MEASURED — KAGGLE.** `FLOOR_TRAIN_ABS_DLOSS`, `FLOOR_CHUNK_SPEC`, and
  the per-layer `δ_β,i` themselves. This node did not, and was instructed not
  to, produce them — they exist only when `measure_noise_floor(..., device=
  "cuda", delta_param_names="self_attn.beta")` runs on the certified device on
  Kaggle and its returned record lands in `COSTS.md` §4 in place of the two
  `☐ NOT YET MEASURED` placeholders there.
- **BLOCKED, on nothing this node owns.** `V17K_RULINGS.md` ledger row 1 stays
  OPEN until the Kaggle run happens and its record is pasted into `COSTS.md` —
  both outside this node's file list by the task's own instruction.
