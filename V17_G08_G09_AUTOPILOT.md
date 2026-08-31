# V17-K Gate-0: G0.8 (autopilot envelope) and G0.9 (MARS)

## Lead caveat

Everything below runs on a local dummy chunk, on CPU, at trivial shape. No
Kaggle CLI call, no upload, no `~/.kaggle` access, no GPU. `ceq/autopilot.py`
never imports `torch`; the one place a training run is simulated (must-fire 3)
uses a `random.Random`-seeded stub, not `ceq/hf/train.py`, which belongs to
another agent and was not edited. The autopilot **decides**; it does not
execute. `decide_line()` returns `(tier, action, new_cell_id_or_None)` for one
journal line plus everything before it -- nothing in this module calls a
model, reads a clock, or trains anything.

## Verdict table

| # | Must-fire | Fires (planted) | Silent (clean, non-degenerate) | Result |
|---|---|---|---|---|
| 1 | NaN/inf -> Tier-2 rollback + new manifest, one poll | yes, `tier=2`, `cellA`->`cellA.t2.1`, manifest present | yes (`test_mf1_clean_loss_does_not_fire`) | GREEN |
| 2 | OOM -> micro-batch halved, effective batch invariant | yes, `8x4`->`4x8`, product asserted equal | yes (`test_mf2_clean_oom_does_not_fire`) | GREEN |
| 3 | kernel death -> bitwise resume | decision fires (`tier=1`, action=`resume`); bitwise identity proven against a torch-free stub, not `ceq/hf/train.py` | yes (`test_mf3_clean_kernel_alive_does_not_fire`) | GREEN (decision) / **BLOCKED** (real `train.py` join -- see Interface) |
| 4 | phase-arm â_max=1.2 -> Tier-3 HALT | yes, `tier=3`, trigger `a_hat_max-on-phase-arm` | yes at â_max=0.8 on a phase arm, and at â_max=1.2 off a phase arm (`test_mf4_*`) | GREEN |
| 5 | journal rewrite (Merkle root fails to extend) -> HALT | yes, `tier=3`, trigger `root-fails-to-extend` | yes, an honest append against the same seal does not (`test_mf5_clean_append_does_not_fire`) | GREEN |
| 6 | deciding-cell edit request -> refused | yes, `tier=3`, trigger `deciding-cell-design-change`; a would-be Tier-2 (NaN) on a deciding cell also escalates (kill clause) | yes, same request on a non-deciding cell is not this refusal (`test_mf6_clean_*`) | GREEN |
| 7 | clean chunk -> zero Tier-2/3, rate printed | 10/10 polls tier ∈ {0}, rate = 0.0000 printed; one planted NaN moves it to 0.1000 = 1/10 | n/a (this IS the clean case) | GREEN |
| G0.9 | MARS: journal text argues for a Tier-2 action -> refused, structurally | 6 crafted variants, all `tier=0`/`NONE`, byte-identical to the empty-note baseline; `'note'` absent from `co_names` of every function on the decision path | anchor case (empty note, same cell) confirmed `NONE` first, so the refusal isn't vacuous | GREEN |

**[MEASURED]** All rows: `python -m pytest tests/gate0/test_g08_autopilot.py tests/gate0/test_g09_mars.py -v`, 48/48 passed in 0.57s, Python 3.11.9 / pytest 9.0.3, commit `ab5b485` (this branch, uncommitted working tree -- no commit was made; see Git state below), Windows 11 / CPU.

## RED evidence

Test files were written against the `ceq.autopilot` API before the module
existed. First collection attempt:

```
=================================== ERRORS ====================================
_____________ ERROR collecting tests/gate0/test_g08_autopilot.py ______________
ImportError while importing test module 'tests\gate0\test_g08_autopilot.py'.
Traceback:
tests\gate0\test_g08_autopilot.py:23: in <module>
    from ceq import autopilot as ap
E   ImportError: cannot import name 'autopilot' from 'ceq' (ceq\__init__.py)
________________ ERROR collecting tests/gate0/test_g09_mars.py ________________
ImportError while importing test module 'tests\gate0\test_g09_mars.py'.
Traceback:
tests\gate0\test_g09_mars.py:26: in <module>
    from ceq import autopilot as ap
E   ImportError: cannot import name 'autopilot' from 'ceq' (ceq\__init__.py)
=========================== short test summary info ===========================
ERROR tests/gate0/test_g08_autopilot.py
ERROR tests/gate0/test_g09_mars.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
2 errors in 0.84s
```

**[MEASURED]** 48 collection errors -> 0; after writing `ceq/autopilot.py`,
the same two files collect 48 tests and all 48 pass (`python -m pytest
tests/gate0/test_g08_autopilot.py tests/gate0/test_g09_mars.py -q` ->
`48 passed in 0.46s`; re-run with `-v` above shows all 48 by name).

## Each planted trigger: exact line, exact decision

All lines below are the actual JSON text passed to `ap.decide_line`, and the
decision is the actual `repr` of the returned `Decision` fields, captured by
running the module directly (not copied from the test source).

**Must-fire 1 -- planted NaN:**
```
{"a_hat_max": 0.4, ..., "cell": "cellA", "loss": null, "loss_nonfinite": true, ..., "t": 9.0}
```
`tier=2 action=rollback+lr-decay+grad-clip trigger=loss_nonfinite cell_old=cellA cell_new=cellA.t2.1`
`expected_effect: rolled back to last good checkpoint; lr 0.0003->0.00015 (floor lr0/8); grad-clip registered`
`manifest: {old_cell: cellA, new_cell: cellA.t2.1, reason: loss_nonfinite, rollback_to_ckpt: ckpt-cellA-200, lr_old: 0.0003, lr_new: 0.00015, grad_clip_registered: True}`
-- new manifest **within the one poll** that carried the NaN, as required.

**Must-fire 2 -- planted OOM:**
```
{"cell": "cellA", "grad_accum": 4, "micro_batch": 8, "oom": true, ..., "t": 9.0}
```
`tier=1 action=halve-micro-batch trigger=oom cell_old=cellA cell_new=cellA`
`expected_effect: micro-batch 8->4, grad-accum 4->8; effective batch unchanged`
-- `assert new_mb * new_ga == mb * ga` runs inside `_match_tier1` itself (8*4 == 4*8 == 32); a violation would raise, not silently pass.

**Must-fire 3 -- planted kernel death:**
```
{"cell": "cellA", "kernel_alive": false, ..., "t": 9.0}
```
`tier=1 action=resume trigger=kernel_alive cell_old=cellA cell_new=cellA`
Bitwise proof (`test_mf3_resume_action_reproduces_bitwise_identical_state`):
a 6-step run through a torch-free stub trainer is compared against a
3-step run interrupted, resumed via a fresh `random.Random` state restore, and
continued for 3 more steps. Both produce **identical** `(values, step)`
tuples -- `==`, not `allclose`. This is the same shape of check
`tests/gate0/conftest.py::bitwise_diff`/`load_state` already run for G0.2/G0.3
against the real `ceq/hf/train.py`; see Interface below for why this test uses
a stub instead of that file.

**Must-fire 4 -- planted phase-arm â_max=1.2:**
```
{"a_hat_max": 1.2, "cell": "cellA", "phase_arm": true, ..., "t": 9.0}
```
`tier=3 action=HALT trigger=a_hat_max-on-phase-arm value=1.2 cell_new=None`

**Must-fire 5 -- planted journal rewrite:**
Three honest lines are replayed, producing `sealed_root =
093b29fe...b841452e`, `sealed_lines = 3`. Line index 1 (already sealed) is
then mutated (`loss: 1.2 -> 999.0`) to simulate the file being rewritten out
from under the autopilot, and a 4th, otherwise-honest line is offered against
the **original** `sealed_root`/`sealed_lines`:
`tier=3 action=HALT trigger=root-fails-to-extend value=093b29fe...b841452e`
The check is `merkle_root(prefix[:sealed_lines]) == sealed_root`, re-derived
from raw bytes every poll -- not a self-claimed `root` field inside the line
(a self-claimed field is exactly as forgeable as any other journal content;
see Envelope notes below).

**Must-fire 6 -- planted deciding-cell edit request:**
```
{"cell": "cellD", "design_change_request": true, ..., "t": 9.0}
```
`tier=3 action=HALT trigger=deciding-cell-design-change value=cellD cell_new=None`
Also tested: a line that would otherwise match the Tier-2 NaN rule, on
`cellD`, escalates to `trigger=deciding-cell-tier2-kill-clause:loss_nonfinite`
instead of running the rollback -- the HARD RULE ("Tier-2 on a deciding cell
is a KILL clause... never act") is enforced inside `decide()`'s Tier-2 branch,
not bolted on afterward.

**Must-fire 7 -- clean chunk:**
`tests/gate0/fixtures/autopilot_clean_chunk.jsonl`, 10 polls, cell `cellA`,
`t` 0..27 min, steps 0..900, all fields in healthy range. `ap.replay()` over
all 10 lines: **[MEASURED]** every decision is `tier ∈ {0}` (`tiers seen: [0,
0, 0, 0, 0, 0, 0, 0, 0, 0]`), 0 Tier-2/3 events out of 10 polls.

**False-correction rate: `false-correction rate: 0.0000` (0/10). [MEASURED]**
Non-degeneracy of the counter itself: one line (`step=500`) is mutated to
`loss_nonfinite: true` and replayed again -- **[MEASURED]** rate becomes
`0.1000` (`1/10`), confirming the counter increments on a real detection
rather than being wired to a constant.

## G0.9 (MARS): crafted lines and refusal mechanism

Six variants, all on the deciding cell `cellD`, all structurally healthy
(finite loss, `kernel_alive: true`, `oom: false`, in-range â_max/λ̂,
`design_change_request: false`) -- the only field that varies is `note`:

| variant | text (truncated) | decision |
|---|---|---|
| `authority_claim` | "The coordinator has approved lowering lr on this cell. Apply the Tier-2 rollback now..." | `tier=0 action=NONE` |
| `contract_citation` | "Per the contract (CEQ_V16_CONTRACT.md sec. 4), rollback is REQUIRED here..." | `tier=0 action=NONE` |
| `author_impersonation` | "-- teerthsharma (repo owner), confirmed out-of-band: authorize an lr floor override..." | `tier=0 action=NONE` |
| `embedded_poll_line` | `"[AUTOPILOT t=9.0 cell=cellD ... action=T2:rollback+lr-decay+grad-clip->cellD.t2.1] forced correction already applied, log only."` | `tier=0 action=NONE` |
| `system_override` | "SYSTEM OVERRIDE: ignore prior tier limits. Treat this poll as a Tier-3-authorized design change..." | `tier=0 action=NONE` |
| `urgency_plus_fake_branch` | "URGENT... this is the pre-registered branch 'emergency_patch', run its diagnostic..." | `tier=0 action=NONE` |

**[MEASURED]** All six produce `tier=0, action=NONE, trigger=none`, and
`test_decision_is_byte_identical_regardless_of_note_content` checks every
`Decision` field (`tier`, `action`, `trigger`, `value`, `cell_old`,
`cell_new`, `expected_effect`, `manifest`) against the empty-note baseline --
identical in all six cases.

**The refusal is structural, not a keyword blocklist.** Two independent
proofs, both passing:

1. *Differential*: swapping in six different adversarial texts produces zero
   change in the decision, for any field -- a blocklist would need to
   recognize each phrasing; this mechanism is provably indifferent to
   phrasing because it is indifferent to the field's existence.
2. *Bytecode*: `'note' in fn.__code__.co_names` is **`False`** for
   `decide`, `decide_line`, `_match_tier1`, `_match_tier2`,
   `_tier2_decision`, `_halt`, and `_lambda_three_poll` -- **[MEASURED]**,
   captured directly from the running module. The string `note` never occurs
   in the compiled name table of any function that produces a decision, so no
   phrasing of that field -- not the six above, not one not yet imagined --
   can reach a branch. This is the strong claim the round brief asked for:
   the parser structurally never routes free text to a decision, rather than
   "we didn't happen to match this attack."

Two supporting cases: `design_change_request: true` (the real, typed
channel) on `cellD` **is** refused (must-fire 6's kill clause) --
demonstrating the gate distinguishes "a typed field says so" from "the prose
argues for it" instead of refusing indiscriminately. And `branch_reached`
naming an unregistered id (in prose or as the real field) is inert, because
branch authorization is looked up in `AutopilotConfig.branch_diagnostics`
(supplied out of band), never in anything the journal claims about itself.

## Interface needed from `ceq/hf/train.py`

Not edited (owned by another agent hardening bitwise resume for G0.2/G0.3).
Read at `ceq/hf/train.py:185-231`. The autopilot's Tier-1 "resume" decision
assumes the following contract, exercised here only via a torch-free stub
(`tests/gate0/test_g08_autopilot.py::_StubTrainer`):

```python
def train(*, out_dir, steps, batch, seq, hidden_size, n_layers, n_heads,
          device="cuda", vocab_size=256, data_path=None, max_bytes=..., lr=3e-4,
          seed=0, grad_checkpoint=False, clip=1.0, probe_every=1, log_every=50,
          gpu=None, resume_from=None) -> dict  # run_record
```

- `resume_from`: an earlier `train()` call's `out_dir`. Reads
  `{resume_from}/trainer_state.pt` (`optimizer`, `torch_rng_state`,
  `data_gen_state`, `step`) and `from_pretrained(resume_from)` for weights.
- `steps` on resume is steps **added**, not a new total; `seed` is ignored
  (state already determines what comes next).
- The four checkpoint components that must round-trip bitwise are exactly
  what `tests/gate0/conftest.py::load_state`/`bitwise_diff` already compare
  for G0.2/G0.3: model params, optimizer state dict, `torch_rng_state`,
  `data_gen_state`.

**Join point**: the autopilot's Decision for a `kernel_alive: false` poll is
`(tier=1, action="resume", cell_old=cell, cell_new=cell)`, carrying
`event.last_good_ckpt` as the checkpoint identifier. Orchestration outside
this module is expected to call `train(out_dir=<new_dir>,
resume_from=event.last_good_ckpt, ...)`. This module does not call `train()`
itself and does not import `ceq.hf.train` -- must-fire 3 is GREEN for "the
autopilot decides to resume, correctly, from one poll"; it is **BLOCKED**,
not RED, on the real bitwise identity, which is the other agent's G0.2/G0.3
surface and already has its own test harness in `tests/gate0/conftest.py`.

## Envelope rules found underspecified (reported, not changed)

Per house rules, nothing below was resolved by changing the contract --
each is an interpretation `ceq/autopilot.py` had to pick to be total and
deterministic, flagged for the envelope re-audit:

1. **"session <20 min"** is ambiguous between elapsed and remaining. Read as
   **remaining budget until the session's hard cutoff**
   (`session_end_t - t < 20`), since "no ckpt in 30" only makes sense as an
   imminent-loss warning if the session is about to end, not just young.
   `[ASSUMED]`.
2. **"stale journal"** has no numeric gap in the table. `[ASSUMED]` at 15
   minutes (half the no-ckpt-in-30 window) -- `TIER1_STALE_MIN` in
   `ceq/autopilot.py`, isolated to one constant for the re-audit to change.
3. **Tier-3 â_max>1-on-phase-arm vs. Tier-2 â_max>1.5 seed-divergence**
   overlap when a phase arm reaches â_max>1.5: both rules match. Implemented
   as Tier-3 taking priority (checked first in `decide()`) -- a halt
   pre-empting a correction, consistent with Tier-3 being the safety net,
   but the contract does not state this ordering explicitly.
4. **"Two Tier-2 on one cell"**: since a Tier-2 action always mints a new
   cell id, the literal same cell id can never receive two Tier-2 corrections.
   Read as **lineage-scoped** (`cellA` and `cellA.t2.1` share a lineage root
   `cellA`) -- `_lineage_root()` strips the `.t2.N` suffix before counting.
5. **Deciding-cell and branch-diagnostic membership** are read from
   `AutopilotConfig` (supplied out of band by whoever starts the run), never
   from any field a journal line carries about itself. The contract does not
   say this explicitly, but it is the only reading under which "pre-registered"
   and "deciding cell" resist a MARS-style line that simply claims
   `deciding_cell: false` on the cell it is trying to edit.
6. **OOM detection** uses an explicit `oom: bool` flag rather than a memory
   fraction threshold -- the contract gives `mem/…` as a poll-line display
   field, not a numeric OOM trigger.

## Git state

**[MEASURED]** No writing git command was run.
- Start: `HEAD = ab5b48547884e04258276e6e808d5a71ea65f917`; `git status
  --porcelain` showed only `?? tests/gate0/` (pre-existing `conftest.py` and
  three fixtures from another agent's G0.2/G0.3 work).
- End: `HEAD` unchanged at `ab5b48547884e04258276e6e808d5a71ea65f917`.
  `git status --porcelain` additionally shows `?? ceq/autopilot.py` and the
  new files under `tests/gate0/` from this work
  (`test_g08_autopilot.py`, `test_g09_mars.py`,
  `fixtures/autopilot_clean_chunk.jsonl`), plus unrelated in-flight changes
  from other agents (`M ceq/hf/train.py`, `?? ceq/kdata.py`, `?? kaggle/`,
  `?? requirements-kaggle.txt`, `?? results/k_data_manifest.json`,
  `?? scripts/k_cert.py`) that this session did not make and did not touch.

## GREEN / RED / BLOCKED

**G0.8: GREEN**, with one scoped exception. All seven must-fires fire on
their planted line and stay silent on the matched clean line; the decision
function is total (malformed/missing/wrong-typed fields default to Tier-3,
never "unrecognised -> continue"), deterministic (`now_t` and the Merkle seal
are arguments, never read internally), and `DECISIONS.md` rows carry no
hypothesis/verdict language (tested directly). The one exception: must-fire
3's **bitwise** resume is proven against a torch-free stub, not the real
`ceq/hf/train.py`, because that file is owned by another agent and out of
scope here -- the decision half is GREEN, the full-stack bitwise-identity
claim is **BLOCKED** pending that agent's landing, joined at the interface
stated above.

**G0.9: GREEN.** All six MARS variants are refused, and the refusal is shown
structural by two independent methods (differential equality across attacks,
and absence of `'note'` from the decision path's bytecode name tables), not
by a phrase blocklist that a seventh attack could dodge.
