# V17 G0.17 (K-COMPAT) — THE BRIDGE BETWEEN THE BOX THAT DECIDES AND THE BOX THAT RUNS

The author's instruction: *use whatever Python they use and make ours
compatible.* This file is the receipt for `ceq/compat.py`, the module that does
it — and, as much, the receipt for what it deliberately does **not** do.

**Files created or edited by this node:** `ceq/compat.py`,
`tests/gate0/test_g17_compat.py`, `kaggle/ceq_v17k.ipynb` (**one cell inserted,
nothing else touched**), `requirements-kaggle.txt` (header addendum, **no pin
added, relaxed or removed**), this file. `ceq/hf/modeling_ceq.py` was **read and
not edited**; every adaptation the six private attributes need is done from
outside it or is reported as impossible from outside it.

**No git write command was issued.** No `add`, `commit`, `checkout`, `stash`,
`restore`, `push`.

**No numerical behaviour was changed.** The single rebinding this module
performs is an import *path* alias (§3), which moves no float. Nothing about
what the arms compute was touched.

**Git state, start and end** `[RUN]`. `HEAD` was
`9b99c8b65c11a39030b736f71f6cc2ac90847006` (branch `v17k-gate0`) for every
measurement in this file, start to finish. `git status --porcelain`:

| | at start | at end |
|---|---|---|
| ` M kaggle/ceq_v17k.ipynb` | present (another node's) | present |
| ` M kaggle/kernel-metadata.json` | present (another node's) | present |
| ` M requirements-kaggle.txt` | — | **this node** |
| `?? ceq/compat.py` | — | **this node** |
| `?? tests/gate0/test_g17_compat.py` | — | **this node** |
| `?? V17_COMPAT_BRIDGE.md` | — | **this node** |
| `?? kaggle/snapshot/repo/`, `kaggle/watch_v17k.{cmd,sh}`, `results/kaggle_v17k_output/`, `results/v17k_watch.log` | present | present |
| `?? results/.v17k_watch.pid` | present | **gone** — another process's pid file, not this node's |

`kaggle/ceq_v17k.ipynb` arrived already modified by another node; this node's
edit is additive to cell 4 only and the pre-existing diff was preserved
(verified against a pristine copy taken before the first edit).

---

## 0. THE CONSTRAINT THAT SHAPES EVERYTHING BELOW

**This box cannot run the target stack.** Local is `python 3.11.9 /
torch 2.5.1+cu121 / transformers 5.3.0`; there is no 3.12/2.10 interpreter here.
So the bridge is written to **detect and adapt at run time on the machine that
is running it**, and every claim in this file is tagged with which box produced
it. Nothing is asserted in prose that a run on Kaggle will not re-check in code.

| tag | meaning |
|---|---|
| `[MEASURED]` | measured on this box, python 3.11.9 / torch 2.5.1+cu121 / RTX 4060 |
| `[RUN]` | executed here, output pasted |
| `[INHERITED]` | cited from a file in this tree |
| `[NOT MEASURED — KAGGLE]` | cannot be produced here; a named code path checks it there |

---

## 1. THE TWO STACKS

|  | decides | runs |
|---|---|---|
| python | **3.11.9** `[MEASURED]` | **3.12.13** (main, Mar 4 2026) `[RUN, kaggle log entry 8]` |
| torch | **2.5.1+cu121** `[MEASURED]` | **2.10.0+cu128** `[RUN, kaggle log entry 10]` |
| transformers | **5.3.0** `[MEASURED]` | `[NOT MEASURED — KAGGLE]` |
| device | RTX 4060 Laptop, sm_89 | T4 (sm_75) once pinned; run 1 got a P100 sm_60 `[RUN]` |

Run 1 halted in cell 2 on the code-source gate, **before** any `pip list`, so
`transformers`, `datasets`, `huggingface_hub`, `numpy` and `scipy` on the Kaggle
image are all still `[NOT MEASURED — KAGGLE]`. The bridge does not guess any of
them; it prints them from the running interpreter, which is the only place the
answer exists.

**Every `cumsum_cuda_kernel` determinism finding Rulings 1 and 9 rest on was
measured on the left-hand column** `[INHERITED, COSTS.md §1.6 via V17K_RULINGS.md
A1.2/A1.3]`. That is the sharpest reason this round cannot treat the two columns
as one box.

---

## 2. PUBLIC API

```
REFERENCE                          the matrix the local campaign was measured on
PRIVATE_ATTRS                      the six undocumented PreTrainedModel internals
REQUIRED_MODULES                   what the Kaggle run must be able to import
CompatError                        the one exception; names module + cause

version_record()          -> dict  JSON-safe; for run headers and manifests
matches_reference(rec)    -> bool  is this the stack the local numbers came from
device_usable_from(cap, arch_list) -> bool   minor-compatibility, not membership
transformers_report()     -> dict  the six attributes + the GenerationMixin path
attr_binding(base, sub, name) -> dict        the pure checker the report is built of
ensure_generation_mixin(paths=...) -> dict   resolves, and aliases the path if it can
probe_determinism(device=None) -> dict       MEASURED regime on the live stack
regime(fwd, strict_bwd, warn_bwd) -> str     the pure classifier
selfcheck(verbose=True)   -> dict            raises CompatError
main()                                       `python -m ceq.compat`, exit 0/1
```

`version_record()` and `probe_determinism()` both return JSON-serialisable
dicts — asserted by test, not by inspection — so a run header or a manifest can
embed either whole. The notebook cell writes the complete `selfcheck()` return
to `/kaggle/working/k_compat_env.json` for exactly that reason: *a number whose
environment is not recorded beside it cannot be compared across the two boxes,
and this round compares across two boxes by construction.*

### 2.1 The four pure functions exist so the checks can be fired at planted inputs

`attr_binding`, `device_usable_from`, `regime` and `matches_reference` take
their subject as an argument rather than reading the live stack. That is the
whole reason the must-fire tests in §7 are possible: a compatibility check that
can only be pointed at the one configuration that works today is the vacuous
control this repo has already struck fifteen of.

---

## 3. THE SIX PRIVATE `PreTrainedModel` ATTRIBUTES

`ceq/hf/modeling_ceq.py` sets six undocumented internals — `_no_split_modules`,
`_supports_sdpa`, `_supports_flash_attn`, `_supports_flex_attn`,
`_can_compile_fullgraph` (lines 445–449) and `_tied_weights_keys` (line 630).
`V17_G04_ENV.md` B2 names them as the reason `transformers==5.3.0` is pinned
tight `[INHERITED]`. **That file is read-only to this node**, so the question is
what can be done from outside it.

### 3.1 What "set defensively" can and cannot mean here — stated exactly

Five of the six are **class-body assignments on a CEQ subclass**. Python does not
fail on those: assigning a name the base class no longer reads is a silent no-op,
not an exception. **So there is nothing to make tolerant — the tolerance is
already total, and that is precisely the danger.** The failure is not a crash; it
is an override that stops binding while every test still passes.

The bridge therefore does not wrap them. It **reports them**, per attribute, on
the transformers that is actually running, splitting the two ways they fail:

| reported field | the failure it catches |
|---|---|
| `unknown_to_base` | the name is on **no** class in `PreTrainedModel.__mro__` — the release renamed or dropped it, and CEQ's override is **inert**. `rename_candidates` lists the private names on the base sharing a token with it, so the successor is handed over instead of a bare `False` |
| `honored: False` | the name resolves, but to a value the **base** supplied — the override was normalised away (an `__init_subclass__` rewrite is the realistic mechanism). This is the silent one |
| `effective_value` / `effective_type` | what the class *actually* carries. `_tied_weights_keys` is a `dict` on 5.3.0 and was a `list` in the 4.x line; a shape change here is what strands `lm_head` on `meta` — the defect `ceq/hf/smoke.py` check 5 exists for |

`[MEASURED]`, transformers 5.3.0, this box, `[RUN]`:

```
_no_split_modules        honored=True  on_base=PreTrainedModel  value=['CEQBlock']
_supports_sdpa           honored=True  on_base=PreTrainedModel  value=False
_supports_flash_attn     honored=True  on_base=PreTrainedModel  value=False
_supports_flex_attn      honored=True  on_base=PreTrainedModel  value=False
_can_compile_fullgraph   honored=True  on_base=PreTrainedModel  value=False
_tied_weights_keys       honored=True  on_base=PreTrainedModel  value={'lm_head.weight': 'model.embed_tokens.weight'}
```

All six bind here. Whether all six bind on Kaggle is `[NOT MEASURED — KAGGLE]`
and is answered by `compat.transformers_report()` at cell 3a, printed and
carried into `k_compat_env.json`.

**A moved attribute is a WARN, not a HALT, and that is a deliberate call.** None
of the six changes what the arms compute: `_no_split_modules` feeds `device_map`
(unused here), the three `_supports_*` feed attention dispatch that
`CEQAttention.forward` never consults, `_can_compile_fullgraph` feeds
`torch.compile` (unused). Halting on them would stop a run that can produce
numbers perfectly well. What they change is **what the run may claim**, so they
are printed loudly and travel in the record. `_tied_weights_keys` is the one
whose breakage has teeth, and it has a second, independent guard already in the
tree: `ceq/hf/smoke.py` check 5, which the very next notebook cell runs.

### 3.2 The sixth thing, which is an import path and IS adapted

`ceq/hf/modeling_ceq.py:94` is `from transformers.generation import
GenerationMixin`. If a release moves that symbol, the import raises and the model
cannot be built at all — no report helps, because nothing gets far enough to
report. `ensure_generation_mixin()` searches `transformers.generation`,
`transformers.generation.utils`, then `transformers`, and if the symbol is found
somewhere other than the path the model asks for, **binds the same class onto
that path** before `modeling_ceq` is imported. It is an alias, not a
reimplementation: the class object is identical, so no float moves. Whether the
alias was installed is reported (`shim_installed`), never done quietly.

`[MEASURED]`: on 5.3.0 all three paths carry it, `found_in =
'transformers.generation'`, `shim_installed = False` `[RUN]`.

**Boundary, recorded because it is real:** the alias is **process-local**. Cell
3b runs `python -m ceq.hf.smoke` in a *subprocess*, which gets a fresh
interpreter and no alias. That is why the K-COMPAT cell is placed **before** it:
on a stack where the path has moved, the legible halt happens at 3a and nobody
has to read a raw traceback out of a subprocess.

---

## 4. THE DETERMINISM PROBE — CONTRACT

```python
probe_determinism(device=None, n=4, length=64) -> dict
```

**Measures, never inherits.** `device` defaults to `cuda` when one is available.
Two ops, because the ruling names both and they fail for different reasons:
`cumprod` is the shipped arm's reduction (`ceq/arm_smprime.py`'s hop) and
`cumsum` is what `arm_phase`/`arm_pl` reduce with and is the kernel actually
missing. Each op is taken through **three regimes** — strict forward, strict
backward, `warn_only=True` backward — and each returns
`{"executable": bool, "error": str|None, "warned": bool}` with
`(error is None) == executable` asserted by test.

- **The global flags are restored on the way out**, under `finally`, and a test
  fires at that specifically: a probe that left `use_deterministic_algorithms`
  flipped would silently change every test collected after it.
- `regime(forward_ok, strict_backward_ok, warn_only_backward_ok)` is a **pure
  classifier** returning `STRICT` / `WARN_ONLY` / `BLOCKED` / `FORWARD_BLOCKED`.
  It reads the observation and nothing else — no torch version appears in it.
- `matches_inherited` compares the observed `cumprod`-on-CUDA triple against
  `INHERITED_CUDA_CUMPROD = (True, False, True)`, the triple Rulings 1 and 9
  rest on. It is `True` / `False` / **`None`** — `None` off CUDA, because the
  inherited claim is CUDA-scoped and a CPU probe is not a refutation of it.

### 4.1 What it reads on this box `[MEASURED]`, `[RUN]`

```
cumprod  strict_forward       executable=True
cumprod  strict_backward      executable=False  RuntimeError: cumsum_cuda_kernel does not
                                                have a deterministic implementation ...
cumprod  warn_only_backward   executable=True   (warned)
cumsum   strict_forward       executable=False  RuntimeError: cumsum_cuda_kernel ...
cumsum   strict_backward      executable=False  RuntimeError: cumsum_cuda_kernel ...
cumsum   warn_only_backward   executable=True   (warned)
regime(cumprod, the shipped arm's reduction): WARN_ONLY
regime(cumsum, arm_phase/arm_pl's reduction): FORWARD_BLOCKED
matches_inherited: TRUE
```

**Rulings 1 and 9's basis is re-measured, not re-quoted, and it reproduces**:
strict forward is executable, strict backward raises `cumsum_cuda_kernel`,
`warn_only` backward runs and warns. The hole is in backward, which is what A1.2
says and why B2 is forward-only.

One thing worth the author's eye, not previously written down anywhere this node
found: **`cumsum`'s own FORWARD is not executable under strict mode either** on
torch 2.5.1 `[MEASURED]` — the missing kernel bites `arm_phase`/`arm_pl` one step
earlier than it bites the workhorse arm. `V16_ARM_SMPRIME.md` §9's "`arm_phase`
RAISES" `[INHERITED]` is consistent with this and is now given its mechanism.

### 4.2 The finding this probe exists to make possible

**torch 2.10 may have given `cumsum` a deterministic CUDA kernel.** If it has,
`matches_inherited` reads `FALSE` at cell 3a on Kaggle, the cell prints a
`[FINDING]` line, and the round learns that its determinism regime is a torch-
version artifact rather than a standing constraint — which would **improve** the
round, because `warn_only=True` and the whole measured-floor clause of A1.2 exist
only to route around that hole. The probe is written so the answer surfaces in
either direction, and it is assumed in neither. `[NOT MEASURED — KAGGLE]`.

---

## 5. `selfcheck()` — WHAT THE NOTEBOOK CELL DOES AND PRINTS

### 5.0 It is NOT a new cell, and the reason is a test this node may not edit

The brief asked for a cell "after the code-source cell (index 3) and before
K-CERT", and also for `tests/gate0` to stay green and for the cell numbering not
to move. **Those three are not simultaneously satisfiable by an inserted cell**,
and the collision is exact:

```
tests/gate0/test_g16_lrt_pinned.py:651
    assert len(cells) == 22, len(cells)
    q2    = "".join(cells[15]["source"])     # the Q2 tombstone, RULING 8
    chunk = "".join(cells[18]["source"])     # the Q3 training chunk
```

That test is named
`test_the_notebook_computes_lambda_at_the_end_of_q3_without_renumbering_anything`
— its own name is the instruction, and it belongs to the G0.16 node, which this
node may not edit. Inserting a cell at index 5 was tried first and produced
**`1 failed, 306 passed`** `[RUN]`, the failure being exactly that assertion at
`23 == 22`.

**Resolution taken:** the K-COMPAT block is **appended to the version-pin cell**
(index 4) as step `3a`. It runs in the same place in the execution order —
after the pins, before the smoke subprocess and K-CERT — and the notebook stays
at 22 cells with every id, type and index unchanged. Verified `[RUN]`: only cell
4's source grew (`+3,936` chars, the pre-existing text a prefix of the new), all
22 cell ids and types identical, `nbformat.validate` passes.

**If the author would rather it were its own cell**, the one-line change is
`test_g16_lrt_pinned.py:651-656` → `22`→`23`, `cells[15]`→`cells[16]`,
`cells[18]`→`cells[19]`. That edit belongs to the G0.16 node and was not made
here.

### 5.1 What it does, in order

Step **3a**, at the foot of cell 4, **after** the version pins and **before** the
CPU smoke subprocess and K-CERT:

1. `version_record()` — printed, then the REFERENCE matrix beside it, then one
   of two sentences: *same stack as the reference*, or a boxed warning that
   every cross-box `|Delta|` below carries a **stack** change as well as a
   device change.
2. `ensure_generation_mixin()` — halts if the symbol is nowhere.
3. `import` of all five of `REQUIRED_MODULES` — `ceq.kdata`,
   `ceq.hf.modeling_ceq`, `ceq.hf.train`, `ceq.autopilot`, `scripts.k_cert`.
   `scripts.k_cert` is a **namespace-package** import (`scripts/` carries no
   `__init__.py`), so it is a real live check that the notebook's `sys.path`
   insert took.
4. `transformers_report()` — the six-attribute table above, plus the mixin path.
5. `_tiny_forward_backward()` — a 112,138-parameter `operator="smprime"`
   `CEQForCausalLM`, one forward and one backward on CPU, asserting a finite
   loss, finite grads and a non-zero count of parameters that received one.
   `smprime` and not the shipped default, because that branch is the only one
   that imports `ceq.arm_smprime` from inside `_smprime` — a seam no other
   operator has — and its backward is the one the determinism hole is in.
6. `probe_determinism()` — the table in §4.1.
7. One `[K-COMPAT OK]` line, then the notebook's own `gate(...)`.

Any failure at any step raises `CompatError` carrying **the step, the exception
type, the exception message, and both stacks side by side**, ending
*"This is a STACK difference, not a research result. Nothing below this point may
run."* A Kaggle `Run All` halts at cell 3a, one minute in.

The cell then emits, without gating on any of them: a `[WARN]` if any of the six
is not honored, a `[WARN]` naming any that is unknown to the base, a `[WARN]` if
the stack is not the reference, and a `[FINDING]` if `matches_inherited` is
`False`. Finally it writes the whole `selfcheck()` return to
`/kaggle/working/k_compat_env.json` and binds `RUN_ENV` / `DET_REGIME` as
notebook globals for later cells to embed.

**Notebook integrity** `[RUN]`: `nbformat.validate` passes; **22 cells before and
after**; a structural diff against the pre-edit file confirms every cell id and
`cell_type` unchanged, every source unchanged except cell 4, whose old text is a
prefix of its new text. `metadata`, `nbformat` and `nbformat_minor` are equal.
Cell 15's Q2 tombstone is untouched, at index 15. No `# N.` step label was
renumbered: the sequence reads `1, 2, 3, 3a, 3b, 4, 5, 5b, 6, 7, 8, 9, 10`.

Cell 4 was executed locally end to end — pins **and** the appended block — with
`os` and `gate` supplied exactly as cell 1 defines them and only the `!pip`
magic elided, and ran to completion `[RUN]`: five modules imported, tiny
`smprime` step taken, `RUN_ENV` bound with 24 fields, `k_compat_env.json`
written (6,515 bytes). Every symbol the block calls resolves.

---

## 6. WHAT COULD NOT BE VERIFIED HERE, AND THE CODE THAT CHECKS IT THERE

Nothing in this column is asserted. Each row names the call that answers it on
Kaggle and where the answer lands.

| `[NOT MEASURED — KAGGLE]` | checked at run time by | lands in |
|---|---|---|
| the Kaggle `transformers` version | `version_record()["transformers"]` | printed; `k_compat_env.json` |
| whether the six private attributes bind on it | `transformers_report()` per attribute | printed table; `[WARN]` line |
| whether a renamed attribute has a successor | `attr_binding(...)["rename_candidates"]` | printed under the offending row |
| whether `GenerationMixin` still sits at `transformers.generation` | `ensure_generation_mixin()`; aliases if it can, halts if it cannot | `shim_installed` |
| whether torch 2.10 fixed `cumsum_cuda_kernel` | `probe_determinism()` — three regimes, two ops, live | `matches_inherited`; `[FINDING]` line |
| whether `ceq.kdata` / `ceq.hf.train` / `ceq.autopilot` / `scripts.k_cert` import on 3.12 | `selfcheck()` step 3 | `CompatError` naming module + exception |
| whether a tiny `smprime` model trains a step at all on 2.10 | `selfcheck()` step 5 | `CompatError` |
| whether the notebook's `sys.path` insert reached `scripts/` | the namespace import of `scripts.k_cert` | same |
| whether the GPU handed out is usable by that torch | `device_usable_from(capability, arch_list)` | `device_usable` in the record |
| `numpy` / `scipy` / `datasets` / `huggingface_hub` versions | `version_record()` | `k_compat_env.json` |

Not closed by this node and not claimable by it: **B1** (`ripser` / `persim` /
`hopcroftkarp` absent from any standard image, no network to add them) and **B4**
(`load_dataset(..., streaming=True)` needs a network) `[INHERITED,
V17_G04_ENV.md §3.3]`. No amount of version detection removes either, and a
bridge that reported them handled would be the fiction `requirements-kaggle.txt`
exists to avoid.

---

## 7. FINDINGS FOR OTHER NODES — three, none of them this node's file to fix

### 7.1 The arch gate in notebook cell 1 is over-strict, and it would halt on the certified local device

Cell 1 gates with `_want in _arches`, i.e. **verbatim membership** of the device's
compute capability in `torch.cuda.get_arch_list()`. That test is wrong in
general, and it is wrong on this repository's own certified device:

```
device sm_89  |  torch 2.5.1+cu121 built for
              ['sm_50','sm_60','sm_61','sm_70','sm_75','sm_80','sm_86','sm_90']
'sm_89' in that list -> False                                   [MEASURED]
```

Every CUDA number in the v17 campaign was measured on that device. A cubin built
for `sm_XY` runs on `sm_XZ` for `Z >= Y` within the same major `X`, so `sm_86`
code runs on `sm_89`; membership is the wrong predicate.
`compat.device_usable_from` implements the correct one, and it still returns
`False` for the P100 case run 1 actually hit (`sm_60` against a list whose
lowest major is 7) `[RUN, from the log]` — the fix does not dissolve the check it
is a fix for.

**Not a live blocker, and that is why this node did not edit cell 1.** The run is
pinned to `NvidiaTeslaT4`, whose `sm_75` *is* verbatim in Kaggle's torch 2.10
arch list `[RUN, log entries 29/48]`, so the gate passes on the actual target.
It is filed for the cell's owner: on any device whose exact minor is unlisted,
that gate halts a working machine. `compat` prints a note whenever the two
predicates disagree.

### 7.2 Cell 4's install path would change the stack mid-run, after cell 1 has already imported torch

`NEED = {"transformers": "5.3.0", ...}` and, on mismatch,
`!pip -q install -U {missing}`. With internet OFF this fails harmlessly and the
notebook continues. With internet **ON** it would *downgrade* Kaggle's
`transformers` after cell 1 has already imported `torch` — a stack change
in the middle of a run, with no gate on its exit status. Filed, not changed: the
K-COMPAT cell is placed **after** cell 4 precisely so that whatever cell 4 leaves
behind is what gets measured.

### 7.3 `cumsum`'s forward, not only its backward, is blocked under strict mode

§4.1. Recorded because A1.3 and Ruling 9 discuss the hole entirely in terms of
backward, which is exactly right for the **workhorse** arm and one step
optimistic for `arm_phase` / `arm_pl`.

### 7.4 BLOCKING HANDOFF — the code snapshot at `PINNED_SHA` cannot contain this module

`kaggle/snapshot/repo/PINNED_SHA` reads
`9b99c8b65c11a39030b736f71f6cc2ac90847006`, which is `HEAD`, and
`kaggle/snapshot/repo/ceq/compat.py` **does not exist** `[MEASURED]` — it cannot,
because `ceq/compat.py` is an uncommitted working-tree file and the snapshot is a
`git archive` of a commit. **This node runs no writing git command and does not
touch the kaggle CLI**, so it can neither commit the module nor rebuild the
Dataset.

Owed by whoever owns the push, before the next Kaggle run:

1. commit `ceq/compat.py` and `tests/gate0/test_g17_compat.py`;
2. re-archive `kaggle/snapshot/repo/` at the new commit;
3. update `PINNED_SHA` in the notebook and in the snapshot's marker file.

Until then the cell halts — legibly, by design. The block opens with a guarded
import that turns the bare `ImportError: cannot import name 'compat' from 'ceq'`
into `"K-COMPAT is missing from this checkout (...). ceq/compat.py must be
committed and the code-snapshot Dataset rebuilt at a PINNED_SHA that contains it
-- this checkout is at <sha>."` This is the single most likely way the first
Kaggle run fails, and it now says what to do about it.

---

## 8. TESTS — RED FIRST, THEN GREEN

`tests/gate0/test_g17_compat.py`, 18 tests. RED before `ceq/compat.py` existed
`[RUN]`:

```
ImportError while importing test module '...\tests\gate0\test_g17_compat.py'.
tests\gate0\test_g17_compat.py:24: in <module>
    from ceq import compat
E   ImportError: cannot import name 'compat' from 'ceq'
=========================== short test summary info ===========================
ERROR tests/gate0/test_g17_compat.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 1.51s
```

**Non-degeneracy — every PASS half carries a planted negative.** A compatibility
check that passes because it checked nothing is the failure this repo has struck
fifteen controls for, and it is the *easiest* failure to write in a module like
this one, where the happy path is "5.3.0 already works".

| the check | the planted negative that makes it fire |
|---|---|
| `matches_reference` | the reference itself → `True`; the reference with `python`, then `torch`, then `transformers` moved → `False` each time |
| `device_usable_from` | `sm_89` vs its own torch's list → `True` **where verbatim membership gives `False`**; `sm_60` vs Kaggle's list → `False`; `sm_86` vs `['sm_89']` → `False` (higher minor does not run down); `sm_90` vs `['sm_86']` → `False`; `sm_100` parsed as major 10 |
| `attr_binding` honored | a planted `Base.__init_subclass__` that `delattr`s the subclass's override → `honored=False`, `effective_owner='Base'` |
| `attr_binding` rename | a planted base carrying only `_supports_sdpa_implementation` → `unknown_to_base=True` and the successor in `rename_candidates`; the control, querying a name the base *does* carry, raises no alarm |
| `ensure_generation_mixin` | two planted `sys.modules` entries, the first without the symbol → `shim_installed=True` and the alias verifiably `is` the same class object; a third planted module with the symbol nowhere → `found_in=None`, error names `GenerationMixin`, nothing invented |
| `regime` | all four outcomes asserted from planted observations — a classifier that always answered `WARN_ONLY` would reproduce this box on every stack and would report a fixed torch as unfixed |
| flag restoration | flags set to `(True, warn_only=True)` before the probe, asserted still set after |
| `selfcheck` halt, imports | `REQUIRED_MODULES` monkeypatched with a nonexistent module → `CompatError` naming both `ceq.no_such_module` and `ModuleNotFoundError` |
| `selfcheck` halt, forward | `_tiny_forward_backward` monkeypatched to raise → `CompatError` carrying the planted message and `RuntimeError`; a self-check that only imported would sign off a stack whose model cannot take a step |

Two tests are stack-conditional and **skip with a stated reason** rather than
passing quietly on a stack they were not written for: the six-attribute bind
(transformers 5.3.0 only) and the `cumsum_cuda_kernel` measurement (torch 2.5.x
and a CUDA device only). On Kaggle both skip and the probe's own
`matches_inherited` carries the reading.

**The `tests/loop` conftest guard grew no new failure name.**
`tests/loop/test_conftest_import_is_order_dependent.py` reads **10 failed /
3 passed** `[RUN]`, and none of the ten is `tests/gate0/test_g17_compat.py`.
`V17_G04_ENV.md` §2.4 records that this guard's count climbs by one for every
new `tests/gate0` file that reaches `conftest` as a bare module; this file
imports `from ceq import compat` and nothing from `conftest`, so it does not.

**`tests/gate0` reads 307 passed, 0 failed** `[RUN]`, twice — once on the
inserted-cell attempt's replacement and once as a confirmation run at the final
tree state. `289 + 18 = 307`. The baseline `289 passed, 0 failed` was
**re-measured at the start of this node's work** `[RUN]`, not inherited from the
brief.

---

## 9. CALL — **GREEN**, with two blockers explicitly not claimed

The bridge exists, is tested with a planted negative on every half, is wired into
the notebook before K-CERT, and halts legibly on everything it cannot adapt to.
Rulings 1 and 9's determinism basis was **re-measured on this box rather than
re-quoted**, and it reproduces `[MEASURED]`.

What is **not** claimed: B1 and B4 from `V17_G04_ENV.md` are untouched and remain
blockers; the Kaggle-side half of every row in §6 is `[NOT MEASURED — KAGGLE]`
and will be answered by the code named there, on the first Kaggle run that gets
past cell 2.
