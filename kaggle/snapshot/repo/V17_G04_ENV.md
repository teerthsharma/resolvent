# V17 G0.4 (K-ENV) — PINNED ENVIRONMENT, AND THE FIVE SUITES MEASURED UNDER IT

Item, verbatim from the contract:

> **G0.4 K-ENV**: pip freeze pinned; suites green under it (arm_smprime 35,
> arm_pl 17, beds 28, certs 36, x35/x35p 43).

Box: Windows 11, Python **3.11.9**, `torch 2.5.1+cu121`, RTX 4060 Laptop (8 GB),
`torch.cuda.is_available()` **True**, `torch.get_num_threads()` **20**
`[MEASURED]`. Every run below was taken at **20 threads**; none re-set the
count. The repo's documented thread floor of `2.345e-3` `[INHERITED, V15/V16
lineage]` is a floor on a *measured quantity*, not on a pass/fail count, and no
suite below reports it — but the thread count is recorded on every run because
in this repo it moves numbers.

**Nothing was trained.** No `--steps` run, no model fitted, no `results/`
journal written by this node. This node ran `pytest` and read package metadata;
that is all.

**No git write command was issued.** No `add`, `commit`, `checkout`, `stash`,
`restore`.

Files this node created: `requirements-kaggle.txt`, `V17_G04_ENV.md`. Nothing
else was edited. No test was adjusted to reach a number.

---

## LEAD CAVEAT — THIS CERTIFICATE WAS WRITTEN AGAINST A MOVING TREE, AND IT MOVED

Five other agents were editing this repository while these runs were taken. The
tree changed **four times inside a twelve-minute window**, and the change is
visible in the `git status --porcelain` recorded before each run:

| clock | `git status --porcelain` | what appeared |
|---|---|---|
| 21:22:44 | *(clean)* | — |
| 21:23:39 | `?? tests/gate0/` | the Gate-0 test directory |
| 21:28:04 | `?? ceq/kdata.py` | the data-loader agent's module |
| 21:31:20 | `?? scripts/k_cert.py` | the device-certification agent's script |
| 21:32:42 | `?? results/k_data_manifest.json` | the data-loader agent's manifest |

By 21:38:32, **after the last measurement in this document**, the same command
also reported ` M ceq/hf/train.py`, `?? ceq/autopilot.py` and `?? kaggle/`
`[MEASURED]`. Those three are *not* covered by any run here: every number below
was taken before they existed. `ceq/hf/train.py` in particular is a **tracked
file with uncommitted modifications** — the first tracked-file edit seen in this
window — and B2/B4 in §3.3 read it as it stood at 21:34, not as it stands now.

`HEAD` was `ab5b48547884e04258276e6e808d5a71ea65f917` for **every** run in this
document, start to finish `[MEASURED]`. Nothing was committed under this node.
All perturbations during the measurement window were **untracked working-tree
files**.

**One of those perturbations changed a suite result while this node was
measuring it, between two runs of the same command.** §2 has it. A `tests/loop`
figure from this document is meaningful only with its timestamp attached, and
every one carries one.

Following the precedent this repo already set — `V16_R1_DEVICE_READY.md` §0
distinguished a *count* change from a *set* change and said which it was
looking at — §2 states, before any number, which of the two it is reading.

---

## 1. THE FIVE NAMED SUITES — EXPECTED vs ACTUAL

Command form, identical for every row, as the repo uses elsewhere:

```
python -m pytest <dir> -q -p no:randomly
```

Taken at `HEAD=ab5b485`, threads 20, 21:22:44–21:24:54 `[MEASURED]`.

| suite | contract expects | **actual** | delta | outcome | attribution |
|---|---|---|---|---|---|
| `tests/arm_smprime` | 35 | **52 passed** | **+17** | all green | contract figure **stale** |
| `tests/arm_pl` | 17 | **17 passed** | 0 | all green | — |
| `tests/beds` | 28 | **28 passed** | 0 | all green | — |
| `tests/certs` | 36 | **36 passed** | 0 | all green | — |
| `tests/x35` + `tests/x35p` | 43 | **58 passed** | **+15** | all green | contract figure **stale** |

**Zero failures across all five, on two independent runs.** The confirmation run
at 21:32:42–21:34:26, with three more untracked files present, collected all six
directories in one invocation: **191 passed in 100.13 s** `[MEASURED]`.
`52 + 17 + 28 + 36 + 14 + 44 = 191`, so the split reproduces exactly.

Per-directory wall clock, first run `[MEASURED]`: arm_smprime 6.31 s, arm_pl
2.27 s, beds 13.82 s, certs 7.28 s, x35 3.72 s, x35p 75.18 s; x35+x35p together
72.14 s.

### 1.1 Both deltas are ADDED TESTS, not a regression — and here is the file

Every collected test passes, so neither delta can be a regression; a regression
shows as a `FAILED` line and there are none. Both deltas are new test files
whose counts were never folded into the ledger the contract's figures came from.
Per-file collection counts, `--collect-only` `[MEASURED]`:

| file | tests | in the contract's figure? | landed in |
|---|---|---|---|
| `tests/arm_smprime/test_arm_smprime.py` | **35** | yes — this *is* the 35 | `91b862d` |
| `tests/arm_smprime/test_runner_wiring.py` | **17** | **no** | **`ab5b485`, the tip commit** |
| `tests/x35/test_residual_onset.py` | **14** | yes | `b0df56e` |
| `tests/x35p/test_kk_crb.py` | **29** | yes — this *is* the 29 | `807e3fe` |
| `tests/x35p/test_source.py` | **15** | **no** | `807e3fe` |

The contract's figures trace to `workdonenewseal.md` §10 "TEST STATE"
`[INHERITED, workdonenewseal.md:488-494]`:

```
tests/arm_smprime 35 passed CPU  ·  35 passed CUDA, same ids
tests/arm_pl      17 passed
tests/beds        28 passed      (BED-K 15 + BED-1 13)
tests/certs       36 passed      (8 mutants killed)
tests/x35         14 passed
tests/x35p        29 passed
```

`14 + 29 = 43` — the contract's combined x35 figure is that ledger's two rows
added. Both stale figures have the same mechanism:

- **arm_smprime 35 → 52.** `test_runner_wiring.py` (17 tests, 295 lines) landed
  in `ab5b485`, *the current HEAD* — the commit titled "Guard the runner's
  identity point against publishing a whole-corpus zero". The contract text
  predates that commit. **The figure was stale on the day it was written down.**
- **x35p 29 → 44.** `test_source.py` (15 tests) landed in `807e3fe`, the *same
  commit* as `test_kk_crb.py` (29). The ledger row was updated with one of the
  two files and not the other. `29` counted `test_kk_crb.py` alone.

**Ruling: the contract's `35` and `43` are stale. The suites did not regress.**
Correct figures, measured twice at `HEAD=ab5b485`: **arm_smprime 52**,
**x35/x35p 58 (14 + 44)**. `arm_pl 17`, `beds 28`, `certs 36` are exact.

This node does not edit `CEQ_V16_CONTRACT.md`; the correction is filed here for
whoever owns that text.

### 1.2 One thing the `[MEASURED]` tag does not cover

`workdonenewseal.md` §10 also claims `tests/arm_smprime` gives "35 passed CUDA,
same ids". This node did **not** re-measure any suite on CUDA. The runs above
are whatever device the suites select for themselves — `tests/arm_smprime`
printed `cpu` on every probe line it emitted. **A CUDA-side count for these five
suites is NOT MEASURED here.** Device measurement belongs to another agent.

### 1.3 GPU contention

Not a factor, and this is checkable rather than asserted: **contention produces
failures or timeouts, and there were none.** All 191 tests passed on both runs.
The confirmation run at 21:32 is itself the required re-run, taken while three
more agents' files were landing. No suite was re-run to convert a red into a
green, because no suite was ever red.

---

## 2. `tests/loop` — WHICH OF THE TWO THIS SECTION READS

**This section reads the failure SET, at the parametrised node-ID level.** The count
is reported second, and is derived from it.

Baseline, `V16_R1_DEVICE_READY.md` §0 `[INHERITED]`: **15 failed / 522 passed**,
five files:

```
test_conftest_import_is_order_dependent.py            × 10
test_corpus_is_recoverable_and_verifiable.py          ×  2
test_every_boundary_node_can_propagate.py             ×  1
test_manifest_refuses_an_absence_it_has_not_earned.py ×  1
test_the_bar_control_is_scored_out_of_sample.py       ×  1
                                                      ---- 15
```

`workdonenewseal.md` §10 gives the same set at **15 failed / 523 passed**
`[INHERITED]` — the pass count had already moved once before this node ran.

### 2.1 Measured

| # | clock | `git status --porcelain` | failed | passed | wall |
|---|---|---|---|---|---|
| 1 | 21:26:22 | `?? tests/gate0/` | **16** | 523 | — |
| 2 | 21:27:05 | `?? tests/gate0/` | **16** | 523 | 35.03 s |
| 3 | 21:31:20 | `?? ceq/kdata.py`, `?? scripts/k_cert.py`, `?? tests/gate0/` | **17** | — | 73 s |

All `[MEASURED]`, all at `HEAD=ab5b485`, threads 20.

### 2.2 The set diff — TWO NEW FAILURE NAMES, both attributed

Sorted `FAILED`/`ERROR` lines, run 1 diffed against the §0 baseline:

```
+ FAILED tests/loop/test_conftest_import_is_order_dependent.py::
      test_no_test_file_imports_conftest_as_a_bare_module[tests/gate0/test_g02_resume.py]
```

Run 3 diffed against run 1 — same command, same HEAD, **five minutes apart**:

```
+ FAILED tests/loop/test_conftest_import_is_order_dependent.py::
      test_no_test_file_imports_conftest_as_a_bare_module[tests/gate0/test_g03_persist.py]
```

The fourteen other baseline lines are **identical, line for line**. No baseline
failure disappeared. The five failing *files* are the same five.

**So: at file level the set is UNCHANGED. At node-ID level it GREW by two, and
both new names are new parametrisations of an already-failing test.** A new
failure name is the thing that matters, and there are two — so this is reported
as a set change, not waved off as a count change.

### 2.3 Attribution — NOT MINE, and here is the assertion

Both new names come from `tests/gate0/`, the **untracked** directory another
agent created at 21:24 and was still writing at 21:27 (`ls -la` mtimes:
`conftest.py` 21:24, `test_g02_resume.py` 21:26, `test_g03_persist.py` 21:27).

The offending lines:

```
tests/gate0/test_g02_resume.py:22:from conftest import bitwise_diff, load_state
tests/gate0/test_g03_persist.py:26:from conftest import bitwise_diff, load_state
```

The assertion that fires, `tests/loop/test_conftest_import_is_order_dependent.py:208`:

```
AssertionError: tests/gate0/test_g02_resume.py reaches conftest as a top-level
module: ['from conftest import bitwise_diff, load_state']. With only 2 of 26
test dirs carrying __init__.py, that name resolves by collection order.
```

`tests/gate0/conftest.py` lines 3–4 say the import form was chosen deliberately,
citing `tests/chase/conftest.py` as precedent. That precedent is itself one of
the ten standing baseline failures — the new files inherit a form the repo
already has a must-fire test against. **This node does not fix it**: it is the
checkpoint/resume agent's file and the checkpoint/resume agent's call. Reported,
attributed, untouched.

### 2.4 The mechanism, which is a finding in its own right

`BARE_CONFTEST_IMPORTERS` is built by scanning the tree **at collection time**.
The parametrisation list — and therefore the failure count — is a function of
what exists on disk the instant pytest starts. `test_g03_persist.py` existed at
21:27 but the run that started at 21:26:22 had already fixed its list, so the
file was invisible to run 1 and failing in run 3. A confirming narrow run of
`tests/loop/test_conftest_import_is_order_dependent.py` alone at 21:28:04 gave
**12 failed / 3 passed**, where the full run 90 seconds earlier gave 11 failures
in that same file `[MEASURED]`.

**`tests/loop`'s failure count is not a stable quantity while another agent is
writing test files.** It climbs by one per new `tests/gate0` file that imports
`conftest` bare. Any Gate-0 sign-off quoting a `tests/loop` count without a
timestamp and a `git status` is quoting noise. Expect 17 or more, not 15.

---

## 3. THE PINNED ENVIRONMENT

Full file with per-line reasoning: **`requirements-kaggle.txt`**.

Method: not `pip freeze` verbatim. This interpreter is shared with other
projects; `pip freeze` returns 300+ distributions, most unrelated — a
`sys.modules` probe during a bare pytest collection here loads `pytest-django`,
`hydra`, `langsmith`, `syrupy` and two editable installs named `faraday` and
`sigmoid`, none of which CEQ imports `[MEASURED]`. The file is instead the
closure of the repo's **actual import statements**, walked with
`grep -rhoE "^\s*(import|from)\s+[a-zA-Z_][a-zA-Z0-9_]*"` over the six suite
directories, followed through the `ceq.*` / `scale.*` modules they reach
(`ceq`, `ceq.beds`, `ceq.certs`, `ceq.certs.topological`, `ceq.hf.modeling_ceq`,
`ceq.x35`, `ceq.x35p`; `scale.arm_a_k1`, `dispatch_count`, `hilbert`,
`journal_scan`, `m3_capability`, `merkle`, `monge`, `planted`,
`r10_capacity_sweep`, `replay_census`, `settle`), then through
`importlib.metadata.requires()` for the two packages with a non-obvious
dependency tail. Versions read with `importlib.metadata.version()` `[MEASURED]`.

### 3.1 The table

**The "in Kaggle base image?" column is `[ASSUMED]` for every row, and the
reason is hard: this node is forbidden the kaggle CLI and has not queried the
image.** It is recorded as an assumption to discharge, not a measurement. What
is *not* assumed is the consequence in §3.2.

| package | version `[MEASURED]` | in Kaggle base image? `[ASSUMED]` | blocker if not |
|---|---|---|---|
| `torch` | 2.5.1 (+cu121 build) | present, **different version** | **HARD** — image ships its own CUDA build; `==2.5.1` unsatisfiable offline |
| `numpy` | 1.26.4 | present, version uncertain | HARD if mismatched |
| `scipy` | 1.17.1 | present, version uncertain | HARD if mismatched |
| `transformers` | 5.3.0 | present, version uncertain | **HARD** — pin is tight for a reason, B2 |
| `pytest` | 9.0.3 | present, version uncertain | soft — any pytest ≥ 7 collects these suites |
| `ripser` | 0.6.14 | **absent** | **HARD** — `tests/certs` (36) loses its oracle |
| `persim` | 0.3.8 | **absent** | **HARD** — same |
| `hopcroftkarp` | 1.2.5 | **absent** | **HARD** — `persim` requires it; nothing else pulls it |
| `Deprecated` | 1.3.1 | likely present | soft |
| `joblib` | 1.5.3 | present | soft |
| `matplotlib` | 3.8.1 | present | soft — needed only so `import persim` succeeds |
| `scikit-learn` | 1.9.0 | present | soft |
| `Cython` | 3.2.4 | present | soft |
| `datasets` | 4.8.4 | present, version uncertain | HARD for the train, not for the five suites |
| `huggingface_hub` | 1.7.1 | present, version uncertain | HARD for the train, not for the five suites |
| `safetensors` | 0.7.0 | present, version uncertain | HARD for the train, not for the five suites |
| `tokenizers` | 0.22.2 | present, coupled to transformers | HARD if transformers is |

Deliberately **not** pinned, reasons in the file: `triton` / `triton-windows`
(nothing in the six suites imports `ceq.mz_kernel` — verified, every other
mention of that name under `ceq/`, `tests/`, `scale/` is inside a docstring;
and the existing `triton-windows` pin carries `sys_platform == "win32"`, which
is False on Kaggle), `gudhi`, `mpmath` (direct), `psutil`, `accelerate`.

### 3.2 The blocker does not depend on knowing the image

With internet OFF, an `==` pin is not a request — it is an **assertion about the
image**, satisfied only when the image happens to carry that exact version. So
the blocking finding holds without a Kaggle probe:

**No exact-pin set derived from a Windows box will be satisfiable by a Linux
Kaggle image except by coincidence, and there is no network to close the gap.**

`requirements-kaggle.txt` is therefore correct as a *record of what was measured
green* and **not** installable as-is on Kaggle. Discharging G0.4 needs a
Kaggle-side `pip list` compared against this table, and then either the pins are
relaxed to what the image carries and the five suites are **re-measured under
those versions**, or a wheel bundle is attached as a Kaggle Dataset (the
standard internet-off route) — which is a separate Gate-0 item nobody currently
owns.

### 3.3 Named blockers

**B1 — `ripser` / `persim` / `hopcroftkarp` are not in any standard scientific
base image.** `ceq/certs/topological.py:397` and `:467` import them
*function-locally*, so a missing package does not break collection — it fails an
assertion deep inside a test, on a machine with no network to fix it. **All 36
of `tests/certs` rest on this.** `hopcroftkarp` is the sharp end: pulled by
nothing but `persim`, obscure, and its absence takes `import persim` down.

**B2 — `transformers==5.3.0` is pinned tight and the pin is load-bearing.**
`ceq/hf/modeling_ceq.py` sets six private `PreTrainedModel` attributes —
verified present in the file at lines **445, 446, 447, 448, 449, 532**:
`_no_split_modules`, `_supports_sdpa`, `_supports_flash_attn`,
`_supports_flex_attn`, `_can_compile_fullgraph`, `_tied_weights_keys`. These are
undocumented HF internals. If the Kaggle image ships a different `transformers`,
there is no network to correct it and no way to know from the outside whether
all six overrides still bind.

**B3 — the repo's only existing "run it elsewhere" recipe assumes internet.**
`colab/train_ceq.ipynb` cell 1 is
`!pip -q install -U 'transformers>=5.0' datasets huggingface_hub accelerate`.
Three faults at once: it needs a network Kaggle will not have; `>=5.0` has no
ceiling and can install a transformers that silently drops one of B2's six
overrides; and `accelerate` is installed but **imported by no module in this
repo** — `ceq/hf/train.py` hand-rolls its loop (`torch.save` of the optimizer
state at line 269), it does not use `transformers.Trainer`. That notebook cannot
be the Kaggle recipe.

**B4 — the train reads its data over the network.** `ceq/hf/train.py:83-84`:
`load_dataset(dataset_id, split=split, streaming=True)`. `streaming=True`
fetches shards at iteration time. **With internet OFF this cannot work at all.**
Outside this node's item — G0.5 owns the loader, and `ceq/kdata.py` plus
`results/k_data_manifest.json` appeared during these runs, presumably as its
answer — but named here because no environment pin can rescue it.

**B5 — the pytest plugin surface differs and is unmeasured.** This box autoloads
`pytest-asyncio`, `pytest-benchmark`, `pytest-cov`, `pytest-django`,
`pytest-mock`, `pytest-timeout`, `xdist`, `syrupy` and `hypothesis` into every
collection `[MEASURED]`. **None is a repo dependency and none is pinned** —
pinning another project's plugins into CEQ's file would be exactly the fiction
this file exists to avoid. But the green counts above were taken *with them
loaded*, and Kaggle's set will differ. Low risk, recorded so it is not a
surprise. Note `pytest-randomly` is **not installed on this box** `[MEASURED]`,
so `-p no:randomly` is a no-op here; it is harmless either way, since pytest
accepts `-p no:X` for an uninstalled `X`.

---

## 4. CALL — **BLOCKED**

Not GREEN and not RED. The two halves of the item split:

**The suite half is GREEN, with a corrected target.** All five suites pass
completely, twice, at `HEAD=ab5b485`, threads 20: **arm_smprime 52 · arm_pl 17 ·
beds 28 · certs 36 · x35+x35p 58 = 191 passed, 0 failed** `[MEASURED]`. Two of
the contract's five figures are stale by exactly the two test files §1.1 names;
nothing regressed and no test was touched.

**The environment half is BLOCKED, on B1–B4.** `requirements-kaggle.txt` is a
faithful record of what was measured green on *this* box. It is not installable
on a Kaggle image with internet off, and B1 (`ripser` / `persim` /
`hopcroftkarp`) plus B4 (`streaming=True`) are blockers no amount of pinning
removes.

**Standing caveat.** Everything above was measured against a tree that changed
four times inside twelve minutes and is still changing. `tests/loop` grew a
second new failure name *between two runs of the same command*. Re-measure
before anything is signed.
