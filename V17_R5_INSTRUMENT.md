# V17-K RULING 5 — the Q1/Q2 instrument, hashed and scope-checked

Discharges `V17K_RULINGS.md` RULING 5: *"`scripts/v15_r1.py` is ADOPTED as the
named instrument. Hash into the identity manifest; one L-SCOPE check (its
batch path == production path); Q1/Q2 cite it by hash."* CPU only. Nothing
trains as a research reading: every `main()` invocation below runs at a
throwaway shape (`n_train ≤ 64`), a throwaway `--tag`, journalled to a `tmp_path`
never `results/`, and exists only so the two files' batch-construction calls
can be captured and diffed. No cell, no NRMSE and no verdict here is a finding.

Repo HEAD at start and at end: `ab5b48547884e04258276e6e808d5a71ea65f917`
(unchanged — no writing git command was run). `git status --porcelain` at
start carried `M scripts/v15_r1.py`, `M ceq/{arm_smprime,hf/*}.py` and a set of
untracked `V17_*.md`/`ceq/autopilot.py`/`ceq/kdata.py`/`scripts/k_cert.py`
files from concurrent nodes; at end the same set plus this node's own edits
(`M scale/identity_manifest.py`, `M scripts/v15_r1.py`,
`tests/gate0/test_g14_instrument_hash.py`) and further untracked files other
nodes wrote while this one ran (`MODEL_CARD.md`, `COSTS.md`,
`tests/gate0/test_g13_beta_learnable.py`, more `V17_*.md`, `scripts/k_cost.py`,
`house-events.jsonl`) — none touched by this node.

---

## LEAD CAVEAT

**The rename is not taken.** `V17K_RULINGS.md` calls it optional and says the
hash is the identity either way; this node agrees with the ruling's own
suggestion to skip it — a rename churns every reference to
`scripts/v15_r1.py` across `V15_R1.md`, `V16_R1_DEVICE_READY.md` and this file
for no gain the ruling asks for, and the hash below is unaffected by the
file's name.

**`r10_capacity_sweep.py` never constructs `arm_pl` or `arm_smprime`.** Its own
docstring says so (`Arm(str)` only), so "batch path == production path" is
checked at the level the two files actually share: the batch construction
(`M3_TASKS`) and the one arm both can run (`softmax`). That is not a narrowing
to dodge the check — `arm_pl`/`arm_smprime` are new constructions this round
added and were never claimed to be on a path `r10_capacity_sweep.py` walks.

**`instrument_hash` names the SCRIPT, not the cell.** It is a new, separate
field from the per-cell `manifest` (`config`/`code`/`shapes`/`rng`) that
`identity_manifest.manifest` already produces for every trained cell.
The two answer different questions and neither replaces the other.

---

## VERDICT TABLE

| item | verdict | evidence |
|---|---|---|
| (a) instrument hashed into identity manifest | **DONE** | `scale/identity_manifest.py:171` `instrument_manifest()`, additive; §1 |
| (b) hash moves when the instrument moves (planted edit) | **HOLDS** | §1.2, one-character plant on the real file |
| (c) what the hash covers is stated, incl. the byte-only gap | **DONE** | §1.1, §1.3 (helper-edit plant) |
| (d) L-SCOPE: batch path == production path | **HOLDS** | §2, `scripts/v15_r1.py:670,679` vs `scale/r10_capacity_sweep.py:341,350`, same-seed run, bit-identical |
| (e) device handling parity | **HOLDS, ONE FORM DIFFERENCE NOTED** | §2.3 — `device="cpu"` vs `device=None` resolve to the identical tensor on cpu |
| (f) Q1/Q2 cell records cite the instrument by hash | **DONE, WIRED AT WRITE SITE** | `scripts/v15_r1.py:585` (header), `:801` (cell); §3 |
| (g) TDD RED before GREEN | **DONE** | §4 |
| (h) non-degeneracy on every PASS half | **DONE** | §1.2, §1.3, §2.2, §3 (four planted perturbations, four catches) |
| (i) no regression in `tests/loop` | **HOLDS** | §5 — identical 15/525, same failing IDs as `V16_R1_DEVICE_READY.md`'s baseline |
| **overall** | **GREEN** | |

Current instrument digest, this tree, `python -m scripts.v15_r1` (module import,
no run):

```
hash    77429cdbb37dc0f3a6c5ca59910a6842063e9c721831d6fbc2665fcd26535498   [MEASURED]
file    ae765c756be25e93081b3bbde9d71bb1cb7208a2cbd2dc541106bc1ee30492de   [MEASURED]
reach   4c9c9e2a706dbf4cd885a75f3f0aeeea57c7e0945083c1958bd970bede7f601a   [MEASURED]
n_reaches  15                                                              [MEASURED]
```

---

## 1. THE HASH — `scale/identity_manifest.py::instrument_manifest`

**Additive only.** `scale/identity_manifest.py` is shared machinery
(`scale/m3_quintuple.py`'s integration points are still reported-not-landed,
per its own STATUS note); nothing existing in it was edited. One function was
added, reusing the module's own `_sha` and `_code_fingerprint` — the same
bytecode fingerprint `manifest()` already uses for a cell's `code` component —
rather than a second hashing scheme.

### 1.1 What the hash covers, and what it plainly does not

Two components, combined into `hash`:

- **`file`** — sha256 of `scripts/v15_r1.py`'s own bytes, read off disk.
  Moves on *any* edit to that file, prose included. This deliberately does
  **not** run through `_code_fingerprint`'s docstring-dropping rule: a
  per-cell `code` component drops comments because a comment edit must not
  invalidate a *published number*, but this instrument's own header states
  load-bearing facts about what it measures (four corrected
  `CEQ_V15_CONTRACT.md` clauses, the F1/F4 device threading, the
  `use_deterministic_algorithms` trade-off) — the file component does not try
  to tell documentation from behaviour and moves on either.
- **`reach`** — `_code_fingerprint` over 15 named callables the instrument
  dispatches through in *other* modules: the batch/oracle/feature triple for
  `e3_t2` (`scale/negation_scope.py`), both gated arms' `forward`/`operator`/
  `readout` (`ceq/arm_pl.py`, `ceq/arm_smprime.py`), the control arm's
  `forward` (`scale/m3_capability.py::Arm`), and the guard/bar functions
  between a draw and a verdict (`calibrate_bar`, `bar_verdict`,
  `refuse_cross_device_pool`, `nrmse`, `bootstrap_ci`). Listed at
  `scripts/v15_r1.py:153-159` as `INSTRUMENT_REACHES`.

**What `file` alone would miss, stated because it is the ruling's own
question.** `file` is a hash of exactly one file's bytes; it cannot see an
edit to `scale/negation_scope.py::make_equilibrium_batch` or any of the other
14 reached callables, all of which live elsewhere and are exactly the
functions that decide what a cell measures. §1.3 proves this is not a
hypothetical: two throwaway helper modules, identical except for one
character, produce the identical `file` hash and different `reach` hashes.

**What `reach` itself does not cover, stated rather than assumed.** Only the
15 named callables are walked. A helper one of *them* calls (say, a function
three imports deep inside `torch.nn.functional`, or a helper this node forgot
to list) is invisible — the same limit `manifest()`'s own `callables=`
argument has always had (its docstring: "this module cannot discover
'everything a script calls' without executing it"). `reaches=()` hashes the
file alone and says so via `n_reaches=0` in the returned dict, so a caller can
tell a scoped hash from an unscoped one without re-deriving it.

### 1.2 Non-degeneracy — the planted one-character edit RULING 5 asks for

On the real file, not a copy standing in for it:

```
orig hash     77429cdbb37dc0f3a6c5ca59910a6842063e9c721831d6fbc2665fcd26535498
mutated hash  ce484f144d6f054c4d85edd5d682408567b7a813c4f8fc0e25df4af92085826b
orig file     ae765c756be25e93081b3bbde9d71bb1cb7208a2cbd2dc541106bc1ee30492de
mutated file  9280cf1b3f8a1fd2205aad793c85942d14ca0086c714e8356baeb82f8b9abfcb
reach unchanged: True    (the plant, `BIND_BAR = 1e-6` -> `1e-7`, touches
                          none of the 15 reached callables)
```

`[MEASURED]`, mutated copy discarded after the read, real file never touched.
The companion non-degeneracy leg — an *unedited* copy must **not** move the
hash — is `tests/gate0/test_g14_instrument_hash.py::test_hash_moves_on_a_planted_one_character_edit`,
asserting both directions in one test.

### 1.3 Non-degeneracy — the reach component's reason for existing

`test_file_bytes_alone_are_defeated_by_a_helper_edit_in_another_module`
(`tests/gate0/test_g14_instrument_hash.py:104`): two throwaway one-line helper
modules differing by exactly one character (`return n + 1` vs `return n + 2`)
stand in for "a module the instrument imports and calls" — `scale/**` and
`ceq/**` are outside this node's edit scope, so the demonstration cannot be
built from the real dependency and says so. `scripts/v15_r1.py`'s own path is
held fixed and unedited on both sides.

```
file_only(v15_r1.py)            -- called twice, hash identical            PASS
instrument_manifest(v15_r1.py, reaches=(helper_v1,))["file"]
  == instrument_manifest(v15_r1.py, reaches=(helper_v2,))["file"]          PASS (the instrument did not move)
instrument_manifest(v15_r1.py, reaches=(helper_v1,))["reach"]
  != instrument_manifest(v15_r1.py, reaches=(helper_v2,))["reach"]         PASS (the ONE-character helper edit moved it)
```

This is the receipt for the sentence in RULING 5's brief: *"A hash over bytes
alone is defeated by an edit to a helper in another module"* — shown, not
argued.

---

## 2. THE L-SCOPE CHECK — batch path == production path

**Claim tested:** the batch construction `scripts/v15_r1.py` uses to score a
cell is the same path `scale/r10_capacity_sweep.py` — the script this repo's
other capability cells are actually produced by — uses, at the same seed,
same shapes, same ordering, same device handling. Established by *running
both*, not by comparing names.

### 2.1 The two call sites, cited

| | eval draw (`seed=12345`) | train draw (`seed` = the cell's) |
|---|---|---|
| **`scripts/v15_r1.py`** (the instrument) | `:670` `batch_fn(a.n_eval, S, D, d_model=D_MODEL, seed=12345, device=dev)` | `:679` `batch_fn(a.n_train, S, D, d_model=D_MODEL, seed=seed, device=dev)` |
| **`scale/r10_capacity_sweep.py`** (production) | `:341` `batch_fn(N_EVAL, S, D, d_model=D_MODEL, seed=12345, device=device)` | `:350` `batch_fn(n, S, D, d_model=D_MODEL, seed=seed, device=device)` |

Both resolve `batch_fn` from `M3_TASKS[task]`, imported `from
scale.negation_scope import M3_TASKS` in both files (`scripts/v15_r1.py:107`,
`scale/r10_capacity_sweep.py:74`).

### 2.2 Structural check, then the run

`tests/gate0/test_g14_instrument_hash.py::test_l_scope_shared_registry_object_not_two_copies`:

```
v15_r1.M3_TASKS is r10_capacity_sweep.M3_TASKS is negation_scope.M3_TASKS      True
v15_r1.M3_TASKS["e3_t2"] is r10_capacity_sweep.M3_TASKS["e3_t2"]               True
(v15_r1.S, v15_r1.D) == (r10_capacity_sweep.S, r10_capacity_sweep.D)           (64, 24) == (64, 24)
v15_r1.D_MODEL is r10_capacity_sweep.D_MODEL                                    True  (scale.m3_capability.D_MODEL, one import)
```

Weak on its own — two files could share an object and still call it
differently — so `test_l_scope_a_mismatched_seed_is_caught` proves the
tensor-equality comparison used below is not vacuously true first: two draws
at seeds 0 and 1 differ (`torch.equal` is `False`); two draws at the same
seed agree bitwise (`torch.equal` is `True`, confirming `make_equilibrium_batch`'s
generator is local to the call, not global state a second call could have
advanced — `scale/negation_scope.py:426` `torch.Generator(device="cpu").manual_seed(seed + 777)`).

### 2.3 Both `main()`s, run for real, same seed, diffed by value

`test_l_scope_batch_path_equals_r10_capacity_sweep_production_path` wraps
`negation_scope.M3_TASKS["e3_t2"]`'s batch_fn in a recording spy (via
`monkeypatch.setitem`, restored automatically), then runs:

```
v15_r1.main()             --arms softmax --seeds 0 1 --n-train 64 --n-eval 4096 --steps 0 --tag g14_lscope_v15
r10_capacity_sweep.main()  --t-star 2 --n-train 64 --max-steps 150 --seeds 0 --threads 2 --tag g14_lscope_r10
```

`ROOT` monkeypatched to `tmp_path` on both, so nothing lands in `results/`.
Every recorded call, matched by `(n, seed)` — `calibrate_bar` draws its own
internal batch at `n=a.n_eval` with its *own* default `seed=0`
(`scale/negation_scope.py:1479`, no `seed=` threaded from the cell), which
collides with a genuine `seed=0` train draw on `seed` alone; `n` (`4096` vs
`64`) is what separates them, and the test's `find()` helper says so at the
site:

| draw | `scripts/v15_r1.py` | `scale/r10_capacity_sweep.py` | `torch.equal(x)` | `torch.equal(y)` |
|---|---|---|---|---|
| eval, `n=4096, seed=12345` | `args=(4096,64,24)` `d_model=16` | `args=(4096,64,24)` `d_model=16` | **True** | **True** |
| train, `n=64, seed=0` | `args=(64,64,24)` `d_model=16` | `args=(64,64,24)` `d_model=16` | **True** | **True** |

`[MEASURED]`. Both `x` (the drawn corpus, `[n, s, d_model]`) and `y` (the
label) are bitwise identical between the two files' own real `main()` runs, at
the same seed. This is the finding: **the batch path is the same path**,
empirically, not by name.

### 2.4 Device handling — one form difference, no value difference

Named separately because it is the one field that differs in *form* without
differing in *effect*, and folding it into the table above would be exactly
the "names match" shortcut the ruling forbids:

```
scripts/v15_r1.py            passes device="cpu"   (a.device, always a string — argparse default "cpu")
scale/r10_capacity_sweep.py  passes device=None    (only set to torch.device("cuda") on --device cuda)
```

`scale/negation_scope.py::make_batch`: `dev = device or torch.device("cpu")`
— a truthy string and a bare `None` resolve to the same `.to(...)` target on
a cpu box. Proved, not assumed: redrawn directly through both spellings at a
throwaway shape,

```
fn(8, S, D, d_model=D_MODEL, seed=99, device="cpu")  vs  fn(8, S, D, d_model=D_MODEL, seed=99, device=None)
torch.equal(x) == True, torch.equal(y) == True                                            [MEASURED]
```

This is exactly why §2.3's tensors compared equal despite the literal
argument differing — reported as a form difference worth knowing, not a
divergence in what gets measured.

### 2.5 Scope of the claim, stated once

`scale/r10_capacity_sweep.py --arm` only ever constructs `Arm(str)` —
`softmax`, `pivot_unsigned`, `windowed_signed`, `pivot_signed`
(`scale/r10_capacity_sweep.py:217-219`) — never `arm_pl` or `arm_smprime`.
"Batch path == production path" is therefore established for the batch
construction and the control arm both files share; it says nothing about, and
was never asked to say anything about, whether `ceq/arm_pl.py` or
`ceq/arm_smprime.py`'s *training loop* matches a "production" trainer for
those arms, because no such production trainer exists in this repo —
`scripts/v15_r1.py` is the only place they are ever trained
(`scripts/v15_r1.py`'s own header: *"`r10_capacity_sweep.train_with_checkpoints`
constructs `Arm(str)` internally and cannot be handed an `ArmPL`"*).

**No divergence was found.** Had one turned up here, this section would report
it instead of a pass — that is what the mismatched-seed plant in §2.2 exists
to make credible.

---

## 3. THE CITATION — how a Q1/Q2 cell record carries the digest

Wired at the write sites, not left as a step a future run has to remember:

- **Header** (`scripts/v15_r1.py:585`): `t="header"` now carries
  `instrument_hash=INSTRUMENT_MANIFEST["hash"]`, so a reader with only the
  console log or the header line, no cell reached yet, can already name the
  instrument that will score everything below it.
- **Cell** (`scripts/v15_r1.py:797-801`): every `t="cell"` row's own
  `r.update(...)` call now includes `instrument_hash=INSTRUMENT_MANIFEST["hash"]`,
  alongside the existing per-cell `r["manifest"]` (`config`/`code`/`shapes`/`rng`,
  from `identity_manifest.manifest`). The two fields answer different
  questions: `manifest` says what config/code/shapes/rng produced *this cell*;
  `instrument_hash` says which build of the *script* ran it — a question
  `manifest`'s own `code` component cannot answer, since it fingerprints only
  the callables named at that one call site, not the file dispatching to them.

`INSTRUMENT_MANIFEST` is computed once, at import time
(`scripts/v15_r1.py:160`), from the file and reach set named in §1 — every
record in a run cites the same digest, and a later reader can recompute it
against the tree that produced the run by calling
`identity_manifest.instrument_manifest("scripts/v15_r1.py", reaches=scripts.v15_r1.INSTRUMENT_REACHES)`.

**Proved to survive the JSON round trip**, not merely to exist as a live
Python attribute: `test_cell_records_carry_the_instrument_hash` runs
`v15_r1.main()` at a throwaway shape, reads `results/g14_citation.jsonl` back
off disk into a fresh process's `json.loads`, and asserts every `t="header"`
and `t="cell"` row's `instrument_hash` equals `v15.INSTRUMENT_MANIFEST["hash"]`
— plus a non-collision leg: a manifest of a *different* file (this test
module itself) produces a different hash.

---

## 4. TDD — RED BEFORE GREEN

Before `scripts/v15_r1.py` was touched, `scale/identity_manifest.py` already
carried `instrument_manifest` (written first), so the hash-related tests were
already green; the citation test was the one still red, on the actual missing
feature:

```
tests/gate0/test_g14_instrument_hash.py::test_cell_records_carry_the_instrument_hash
    expected = v15.INSTRUMENT_MANIFEST["hash"]
               ^^^^^^^^^^^^^^^^^^^^^^^
E   AttributeError: module 'scripts.v15_r1' has no attribute 'INSTRUMENT_MANIFEST'

2 failed, 5 passed in 3.79s
```

(The L-SCOPE test failed alongside it on that same run, but for a test-authoring
bug of this node's own — a single train seed left `statistics.stdev` with one
data point — fixed before being counted as evidence; re-run with the fix and
`INSTRUMENT_MANIFEST` still absent, only the citation test was red, confirming
the L-SCOPE finding in §2 does not depend on the wiring in §3.)

After `scripts/v15_r1.py:151-160,585,797-801`:

```
tests/gate0/test_g14_instrument_hash.py::test_instrument_manifest_is_a_pure_read_and_reproducible PASSED
tests/gate0/test_g14_instrument_hash.py::test_hash_moves_on_a_planted_one_character_edit PASSED
tests/gate0/test_g14_instrument_hash.py::test_file_bytes_alone_are_defeated_by_a_helper_edit_in_another_module PASSED
tests/gate0/test_g14_instrument_hash.py::test_l_scope_shared_registry_object_not_two_copies PASSED
tests/gate0/test_g14_instrument_hash.py::test_l_scope_a_mismatched_seed_is_caught PASSED
tests/gate0/test_g14_instrument_hash.py::test_l_scope_batch_path_equals_r10_capacity_sweep_production_path PASSED
tests/gate0/test_g14_instrument_hash.py::test_cell_records_carry_the_instrument_hash PASSED
7 passed in 5.92s
```

**Non-degeneracy, per test, so every PASS half above has a catch behind it**
(this repo has struck 15 vacuous controls):

| test | plant | catch |
|---|---|---|
| `test_hash_moves_on_a_planted_one_character_edit` | 1-char edit to `scripts/v15_r1.py`'s own bytes | `file`/`hash` move; an unedited copy does not |
| `test_file_bytes_alone_are_defeated_by_a_helper_edit` | 1-char edit to a callable in `reaches` | `reach`/`hash` move while `file` (of the unedited instrument) does not |
| `test_l_scope_a_mismatched_seed_is_caught` | seed 0 vs seed 1 | `torch.equal` returns `False` |
| `test_cell_records_carry_the_instrument_hash` | manifest of an unrelated file | different hash (non-collision) |

---

## 5. REGRESSION CHECK

`python -m pytest tests/loop -q -p no:randomly`, this node's edits in place:

```
15 failed, 525 passed, 3 warnings in 48.15s
```

The 15 failing IDs are line-for-line the same set `V16_R1_DEVICE_READY.md` §0
names as the standing baseline (`test_conftest_import_is_order_dependent.py`
×10, `test_corpus_is_recoverable_and_verifiable.py` ×2,
`test_every_boundary_node_can_propagate.py` ×1,
`test_manifest_refuses_an_absence_it_has_not_earned.py` ×1,
`test_the_bar_control_is_scored_out_of_sample.py` ×1) — verified by name, not
only by count. 525 passed here against 522 in that report; the extra 3 are
other nodes' additions to `tests/loop` since, not this node's.
`[MEASURED]`.

`python -m pytest tests/gate0 -q`: **2 additional failures**,
`test_g13_beta_learnable.py::test_the_beta_zero_corner_logits_are_the_pre_change_bytes`
and `::test_the_beta_zero_corner_comparison_catches_one_ulp`. Not this node's:
that file imports only `ceq.arm_smprime`, `ceq.hf.modeling_ceq`,
`ceq.hf.train`, `ceq.hf.configuration_ceq` — none edited here — and
`ceq/arm_smprime.py` was already `M` (modified) in `git status` before this
node's first command, mid-edit by RULING 2's beta node. Reported because it
was seen, not because it is this node's to fix.

`tests/cameron/test_identity_manifest.py`,
`tests/loop/test_identity_manifest_covers_every_beyond_key_field.py`,
`tests/arm_smprime/test_arm_smprime.py`: all pass except the one
pre-baseline failure above; `instrument_manifest`'s addition moved nothing in
`manifest()` or `refuse_if_changed`, which it does not touch.

---

## WHAT THE RE-TAKE NODE (RULING 4) MUST KNOW

1. **No behaviour of the instrument changed.** `main()`'s bind checks, bar
   calibration, training loop, verdicts and printed output are byte-for-byte
   what they were; the only additions are two new fields on records already
   written (`instrument_hash` on `t="header"` and `t="cell"`) and one
   module-level computation at import time (`INSTRUMENT_MANIFEST`, a pure
   read of the file plus 15 callables — no tensor touched, no RNG advanced).
2. **Every cell the CUDA re-take produces will carry `instrument_hash`**
   automatically — nothing to opt into, nothing that can be forgotten.
3. **The digest is a function of the tree, not the device.** Re-running on
   the certified 4060 with this same `scripts/v15_r1.py` produces the
   identical `instrument_hash` computed here (`77429cdb…6535498`) as long as
   the file and its 15 reached callables are unedited between now and then;
   if any of them move, the hash moves with them and a stale citation is
   detectable by recomputing `instrument_manifest` against the tree at hand.
4. **`INSTRUMENT_REACHES` is a python object, not a file to diff.** A reader
   wanting to audit the reach set reads `scripts/v15_r1.py:153-159` directly;
   it is not serialized anywhere the re-take needs to carry forward.

---

## LIMITS

- **The L-SCOPE check is scoped to batch construction and the control arm**,
  because that is the only path `scale/r10_capacity_sweep.py` and
  `scripts/v15_r1.py` actually share (§2.5). It says nothing about whether
  `ceq/arm_pl.py`/`ceq/arm_smprime.py`'s training loop matches some other
  "production" trainer, because none exists to compare against.
- **`reach` covers 15 named callables, not a transitive closure.** A helper
  one of them calls, that nobody listed, is invisible to `reach` exactly as it
  is to `manifest()`'s own `callables=` argument — stated in
  `instrument_manifest`'s docstring, not discovered by a test, because
  discovering it would require executing the file.
- **The device-handling check (§2.4) is one shape (`n=8`), one seed (`99`).**
  It establishes that `device="cpu"` and `device=None` agree at all, not that
  they agree at every shape this repo trains at.
- **`test_g13_beta_learnable.py`'s two failures are reported, not
  investigated.** They belong to `ceq/arm_smprime.py`, outside this node's
  edit scope and mid-flight under a concurrent node at the time this run was
  taken.
- **Nothing here re-derives whether Q1 or Q2 themselves are ready to run.**
  This node priced the instrument's identity and its batch path; it trained
  nothing and adjudicated no cell.
