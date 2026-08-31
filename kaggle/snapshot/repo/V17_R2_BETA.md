# V17-K — RULING 2 / 2a, CODE HALF: β LEARNABLE, INIT 1, LOGGED PER INSTANCE

**LEAD CAVEAT, BEFORE ANY NUMBER BELOW.** This is INSTRUMENTATION and nothing
else. Nothing here was trained as a research reading: the loops in this file run
1–6 gradient steps at a 27,914-parameter shape on random-ish text and report
**only** that `β` moved, that its gradient is finite, and that the logged column
is not stale. **No loss curve, no NRMSE, no cell of R1′/R2/any bed, no seed pool,
no verdict, no comparison to softmax or to anything else.** No number in this
file says the arm is good, bad, or comparable. Every `β` value printed here came
from a probe designed to move a scalar, not to train a model.

**SECOND CAVEAT: β WAS ALREADY LEARNABLE WHEN THIS NODE ARRIVED.** The wiring
node (`V17_ARM_WIRING.md`) had already made `β`, `qk` and `g` `nn.Parameter`s on
`CEQAttention`, initialised from `SMPRIME_CORNER = (1.0, 1.0, 1.0)`. This node
**verified and pinned** that rather than building it, and built the logging,
the census and the pinning arithmetic on top. Item 1 of the brief was
**already discharged by another node** and is reported as such, not claimed.

**THIRD CAVEAT: THE PINNING BRANCH CANNOT BE TAKEN FROM ANYTHING THIS NODE
MEASURED.** RULING 2a's criterion needs `δ_β` from the RULING 1 identical-seed
pair, which is `scripts/k_noise_floor.py`'s output and does not exist yet on the
certified device. `beta_summary` **refuses to branch** without it and says so in
`branch_reason`. The one `delta_beta` printed in section 7 is a **placeholder
for shape only** and is labelled `[ASSUMED]`.

**Files written.** `ceq/hf/modeling_ceq.py` (edited),
`tests/gate0/test_g13_beta_learnable.py` (new), this file. **Nothing else was
touched** — in particular not `ceq/arm_smprime.py`, not `ceq/hf/train.py`, not
`ceq/hf/configuration_ceq.py` (owned but not needed), not `ceq/autopilot.py`,
not `scripts/`, not `scale/`, not `MODEL_CARD.md`, not `V17K_RULINGS.md`. **No
git command that writes was run.**

**Provenance.** `git rev-parse HEAD` = `ab5b48547884e04258276e6e808d5a71ea65f917`
at start **and** at end (no commit made by this node). `git status --porcelain`
at start: 4 modified (`ceq/arm_smprime.py`, `ceq/hf/configuration_ceq.py`,
`ceq/hf/modeling_ceq.py`, `ceq/hf/train.py`) + 12 untracked, all peer-owned. At
end: 8 modified + 27 untracked — the growth is other nodes landing
(`MODEL_CARD.md`, `scale/identity_manifest.py`, `scripts/k_noise_floor.py`,
`V17_R2A_CARD.md`, …); **of those paths this node wrote exactly two**,
`ceq/hf/modeling_ceq.py` and `tests/gate0/test_g13_beta_learnable.py`, plus this
report. Box: Windows 11, RTX 4060 Laptop 8 GiB, torch `2.5.1+cu121`,
transformers `5.3.0`, python 3.11. **Every measurement below is CPU, float32,
`hidden_size=32, n_layers=2, n_heads=4, vocab=32, seq=16, batch=2`, made this
run.** L-TIME: no timing is inherited and none is reported.

---

## VERDICT TABLE

| question | answer |
|---|---|
| **(a) Is β learnable, and at what granularity?** | **YES, and PER LAYER.** `CEQAttention.beta` is an `nn.Parameter`, `shape ()`, `requires_grad=True`, present in `model.parameters()`, at exactly `1.0`. Granularity is one scalar per **layer**, shared across attention heads, because that is what `ceq/arm_smprime.py::ArmSMPrime` has. §3. `[MEASURED]` |
| **(b) Does a gradient actually reach it?** | **YES.** One forward+backward at the shipped corner: `∂L/∂β = 3.1159e-04` (layer 0), `2.7389e-04` (layer 1), both finite and non-zero. The mechanism is `d/dβ Z^β = Z^β·log Z`, and `Z_i ≥ 1` with `Z_i ≠ 1` off the first row. `[MEASURED]` |
| **(c) Is the β=0 corner bit-identical before and after?** | **YES, `torch.equal` on weights AND logits, at all three corners, max\|Δ\| = 0.000e+00.** Pre-change files snapshotted before the first edit, rebuilt as a throwaway package, run side by side in one process. §4. `[MEASURED]` |
| **(d) Is that comparison non-degenerate?** | **YES, two ways.** One float32 ulp on `m_head.weight` breaks it (`max\|Δ\| = 8.941e-08`); and moving `β` from `0` to `1e-6` **changes the (L)-corner logits**, so `β` is INACTIVE at that corner, not algebraically cancelled. §4.2 — this is the check the 15-strike record demanded. `[MEASURED]` |
| **(e) Can a "β never moved" run be told from a "β moved and returned" one?** | **YES, and the final value alone CANNOT.** Three runs whose final β is `[1.0, 1.0]` land in three different states — `immobile`, `not_updated`, `moved` — off the trajectory, the gradient column and the census. §5. `[MEASURED]` |
| **(f) Shape of the logged column** | `{"name": [str]×L, "beta": [float]×L, "grad": [float\|null]×L, "requires_grad": [bool]×L}`, one per log point; plus `beta_census`, `[float]×L`, an integral over **every** step. §7. |
| **(g) RULING 2a implemented?** | **YES: criterion, three branches, quantifier, census word, and the degenerate-floor case.** `beta_summary(series, census=…, delta_beta=…, k=5.0)`. §6. |
| **(h) Does this node's own proposal survive 2a?** | **NO, and the argument against it is made here.** The loss-substitution criterion licenses a STRONGER sentence than 2a in one direction, so by "which claims less" 2a wins. The substitution is **retained as a diagnostic that can only weaken a card sentence, never establish one**. §6.4. |
| **(i) Interface needed from `ceq/hf/train.py`** | **Five lines, exact patch in §8.** Two strict-`xfail` tests hold the seam: the run record must carry `beta` and `beta_census`, and `_OPERATOR_KEYS` must carry the three `smp_*` knobs (it does not, so `build()` **silently drops** them today). |
| **(j) Regression** | `tests/gate0` **253 passed / 2 xfailed** (was 182 passed before this file existed, +32 +2 here and +37 from peers landing mid-run). `tests/arm_smprime` **52 passed**, unchanged. The five HF-package files in `tests/chase`: **4 failed / 120 passed / 2 xfailed**, the same four by name the wiring node recorded, all about the WITHDRAWN decay exponent and the COSTS prose, none about β. §9. |
| **(k) GREEN/RED/BLOCKED** | **GREEN on the code half.** `tests/gate0/test_g13_beta_learnable.py` **32 passed / 2 xfailed**, from **24 failed / 7 passed / 2 xfailed** RED. **The card's branch is BLOCKED on `δ_β`** from the floor node, by construction and not by an omission. |

---

## 1. RED

Tests were written first. The RED transcript below is the **final** test file run
against the **pre-change** `ceq/hf/modeling_ceq.py` — the snapshot taken before
the first edit was copied back over the live file, the suite was run, and the
file was restored and sha256-verified identical
(`b3aa5dfd428becfe0ba017f27792a1b70d8c43baa26a7bcc23ce79ab8ab1a4c9` both sides).
That is stronger than the transcript of the first RED run, because RULING 2a
arrived mid-node and its tests would otherwise have been written after their
implementation; this re-run makes every one of them RED against the tree as it
stood before this node touched it.

```
FAILED tests/gate0/test_g13_beta_learnable.py::test_the_beta_column_is_one_scalar_per_layer_and_tracks_the_parameters
FAILED tests/gate0/test_g13_beta_learnable.py::test_the_beta_column_is_empty_on_an_operator_that_has_no_beta
FAILED tests/gate0/test_g13_beta_learnable.py::test_a_stale_beta_column_is_caught
FAILED tests/gate0/test_g13_beta_learnable.py::test_the_summary_recovers_the_final_distribution_from_the_series
FAILED tests/gate0/test_g13_beta_learnable.py::test_a_frozen_beta_and_a_beta_that_moved_and_returned_have_the_same_final_value
FAILED tests/gate0/test_g13_beta_learnable.py::test_a_frozen_beta_is_distinguished_from_a_beta_that_moved_and_returned
FAILED tests/gate0/test_g13_beta_learnable.py::test_a_detached_beta_gradient_reads_immobile_and_names_itself
FAILED tests/gate0/test_g13_beta_learnable.py::test_a_beta_the_optimizer_never_updates_is_its_own_state
FAILED tests/gate0/test_g13_beta_learnable.py::test_the_three_states_are_pairwise_distinct
FAILED tests/gate0/test_g13_beta_learnable.py::test_beta_actually_moves_on_an_ordinary_run
FAILED tests/gate0/test_g13_beta_learnable.py::test_the_census_is_the_integral_of_the_gradient_over_every_step
FAILED tests/gate0/test_g13_beta_learnable.py::test_the_census_grows_and_a_one_step_census_is_not_a_three_step_one
FAILED tests/gate0/test_g13_beta_learnable.py::test_a_detached_beta_has_a_census_of_exactly_zero
FAILED tests/gate0/test_g13_beta_learnable.py::test_a_frozen_beta_has_a_census_of_exactly_zero
FAILED tests/gate0/test_g13_beta_learnable.py::test_the_criterion_is_five_delta_beta_and_the_boundary_is_inclusive
FAILED tests/gate0/test_g13_beta_learnable.py::test_the_three_branches_are_reachable_and_the_thresholds_are_not_a_dichotomy
FAILED tests/gate0/test_g13_beta_learnable.py::test_branch_c_names_where_the_moved_betas_live
FAILED tests/gate0/test_g13_beta_learnable.py::test_a_bitwise_identical_seed_pair_makes_the_criterion_degenerate
FAILED tests/gate0/test_g13_beta_learnable.py::test_the_census_word_is_the_weaker_one_unless_the_comparison_group_exists
FAILED tests/gate0/test_g13_beta_learnable.py::test_without_delta_beta_the_summary_refuses_to_branch
FAILED tests/gate0/test_g13_beta_learnable.py::test_the_substitution_delta_is_exactly_zero_when_beta_is_already_one
FAILED tests/gate0/test_g13_beta_learnable.py::test_the_substitution_delta_is_nonzero_when_beta_left_one
FAILED tests/gate0/test_g13_beta_learnable.py::test_the_substitution_restores_beta_on_the_way_out
FAILED tests/gate0/test_g13_beta_learnable.py::test_the_substitution_is_blind_to_whether_beta_could_move
24 failed, 7 passed, 2 xfailed, 2 warnings in 8.98s
```

**THE SEVEN THAT PASSED AT RED ARE THE SEVEN THAT WERE SUPPOSED TO, AND EACH IS
A REGRESSION PIN RATHER THAN A CLAIM.** Two assert β is a learnable `nn.Parameter`
at 1.0 with a finite non-zero gradient — work the wiring node had already landed,
so a test of it passes before this node's change by construction. Three pin the
(L) corner against constants measured on the pre-change tree — a regression guard
passes before the regression, which is why each has a **separate** planted
negative and every one of those was RED. Two pin
`ceq/arm_smprime.py::label_cell`'s β=0 certificate, a file this node may not edit.

**The two xfails are STRICT and both are RED for a real reason** — they describe
the `ceq/hf/train.py` seam this node may not write (§8). They will turn into
loud XPASS failures the moment that patch lands, which is the handoff signal.

## 1b. GREEN

```
$ python -m pytest tests/gate0/test_g13_beta_learnable.py -q
32 passed, 2 xfailed, 2 warnings in 8.24s

$ python -m pytest tests/gate0 -q
253 passed, 2 xfailed, 2 warnings in 89.05s
```

---

## 2. WHAT WAS ALREADY THERE, AND WHAT THIS NODE ADDED

`ceq/hf/modeling_ceq.py::CEQAttention.__init__`, **unchanged by this node**:

```python
self.beta = nn.Parameter(torch.tensor(float(config.smp_beta)))
```

with `config.smp_beta` defaulting to `SMPRIME_CORNER[0] = 1.0`. So the LM's
wiring did **not** pin β as a buffer or a constant, and item 1 of the brief
needed no code. It needed **checking**, and the check is asserted on the
GRADIENT rather than on the `requires_grad` flag, because a parameter whose
gradient is identically zero is frozen in every sense RULING 2 cares about.

Added by this node, all in `ceq/hf/modeling_ceq.py`, all pure reads or
list arithmetic, none of them touching a forward path:

| function | what it is |
|---|---|
| `beta_column(model)` | the logged column: name, value, gradient, flag, per layer |
| `beta_census(model, acc)` | integrated `\|∂L/∂β_i\|`, one running total per β |
| `beta_summary(series, census=, delta_beta=, k=)` | the trajectory read back: state, distribution, RULING 2a's criterion, branches, census word |
| `beta_substitution(model, **fwd)` | loss at trained β vs loss at β←1 (a diagnostic, §6.4) |
| `_beta_parameters(model)` | `[(name, parameter)]`, the shared accessor |

The only other edit is `import statistics` (stdlib). **No new construction**: no
buffer on the module, no state_dict entry, no change to the parameter count, no
change to any forward. §4 proves that at three corners with `torch.equal`.

---

## 3. THE GRANULARITY, AND WHY IT IS NOT PER HEAD

`ceq/arm_smprime.py::ArmSMPrime.__init__` carries

```python
self.beta = nn.Parameter(torch.tensor(1.0))
```

**one scalar per arm instance.** `CEQAttention` carries that same
parametrization once per layer. Therefore:

* **the instance is the LAYER**; the column is `[n_layers]`;
* **there is no head axis to report**, and RULING 2a branch C's "which layers /
  heads" is answerable as **layers**. Widening β to `nn.Linear(d, n_heads)` or to
  a `[n_heads]` parameter would be a construction the arm does not have — the
  same argument `V17_ARM_WIRING.md` §2 makes for refusing to widen the gate
  heads — and the round's first law strikes it. It was not built.

The identity is emitted **by name**, not left implicit in the list position:

```
["model.layers.0.self_attn.beta", "model.layers.1.self_attn.beta", ...]
```

which is bitwise the key `named_parameters()` produces
(`test_the_column_name_is_the_key_the_floor_node_will_hand_back`, `[MEASURED]`),
so the join against `scripts/k_noise_floor.py::per_parameter_deltas` is by
identical key and the absence of a head index is visible **in the artifact**
rather than only in this prose.

---

## 4. THE β=0 CORNER, INTACT

### 4.1 Bit-identity, before against after

`ceq/hf/{modeling,configuration}_ceq.py` were copied into a scratch package
**before the first edit** (`modeling_ceq.py` sha256
`76da863de067b295b60f04ad186f2b697406e566ed4a643c5482f0a7beae3c67`), then both
trees were built and forwarded side by side in one process, same construction
seed, same input:

```
(L) beta=0 qk=0 g=1   n_params= 27914  weights torch.equal=True  logits torch.equal=True  max|d|=0.000e+00
SMPRIME_CORNER (1,1,1) n_params= 27914  weights torch.equal=True  logits torch.equal=True  max|d|=0.000e+00
sgate default path     n_params= 27776  weights torch.equal=True  logits torch.equal=True  max|d|=0.000e+00
PLANT 1 ulp m_head.weight             logits torch.equal=False  max|d|=8.941e-08
```

`torch.equal`, not `allclose`. `[MEASURED]`

The certificate the card's **identity clause** actually cites lives one level
down, in `ceq/arm_smprime.py::label_cell` — a file this node may not edit. It is
pinned rather than asserted-by-abstinence:

| quantity | value |
|---|---|
| `label_cell(bedm_draw(seed=15, s=8))["beta"]` | `0.0` |
| `["qk"]` | `0.0` |
| `["residual"]` | `1.1102230246251565e-16` |
| `["residual_bos"]` | `0.0` |
| `["manifest"]["hash"]` | `3994d8f4c0a2f318dced36589c201e9986d8cb89203f0b5484851885c6908ad3` |

`[MEASURED]` on the pre-change tree and again after. The residual and the BOS
residual are pinned in the test; the hash is reported and not pinned, because
`scale/identity_manifest.py` hashes **bytecode** and a peer's edit to a branch in
`arm_smprime.py` would move it for reasons that are not this node's.

### 4.2 Why this is not the sixteenth vacuous control

The obvious objection: at β = 0, `Z^β = 1` for every `Z`, so a certificate read
at that corner might be read through a quantity in which the switch it certifies
**algebraically cancels** — which is exactly the shape of the most recent strike.

It does not cancel. `d/dβ Z^β = Z^β·log Z`, which at β = 0 is `log Z ≠ 0`. The
corner is a point where β is **inactive**, not one where it is **absent**, and
that is measurable:

| build | (L)-corner logits sha256 (1 thread) |
|---|---|
| `smp_beta = 0.0` | `5237c8bdcb17ca793b00ea5e074eac8bb0fe5ca73a946c57094447bf59823a07` |
| `smp_beta = 1e-6` | `4011f7044da2d772dbf3af1329523ab5ca336f88583447edffc6a26a83d8c833` |
| `smp_beta = 1e-4` | `5c96f4a4f8470437db9fb3f4f2def4ac007c8b7e049e44a1ae8ee8db83f5e7fa` |

A **one-part-in-a-million** move of β off the corner changes the fp32 logit bytes.
`[MEASURED]`, `test_the_beta_zero_corner_comparison_is_not_blind_to_beta`.

### 4.3 Two findings the plant search produced, both reported

**(i) The plant had to be moved, and the measurement is here.** One float32 ulp,
per parameter, at the (L) corner, one thread:

```
model.embed_tokens.weight                      caught at   2 ulp  max|d|=5.215e-08
model.layers.0.self_attn.qkv.weight            NOT caught up to 64 ulp
model.layers.0.self_attn.o_proj.weight         caught at   1 ulp  max|d|=4.843e-08
model.layers.0.self_attn.m_head.weight         caught at   1 ulp  max|d|=5.960e-08
model.layers.0.self_attn.m_head.bias           NOT caught up to 64 ulp
model.layers.0.self_attn.theta_head.weight     NOT caught up to 64 ulp
model.layers.0.mlp.0.weight                    caught at   1 ulp  max|d|=5.960e-08
lm_head.weight                                 caught at  16 ulp  max|d|=2.980e-08
```

`m_head.bias` is off target because `_init_weights` zeroes every `nn.Linear`
bias, so one ulp above `0.0` is a **denormal**. `qkv.weight` is off target for a
load-bearing reason: at `qk = 0` the content term is deleted **exactly**, so the
q/k projections are dead weights at that corner — a property of the corner, not
a defect. The plant used is `m_head.weight`, the arm's own magnitude head, caught
at 1 ulp. `[MEASURED]`

**(ii) A BITWISE PIN THAT DOES NOT NAME ITS THREAD COUNT IS ORDER-DEPENDENT.**
The (L)-corner sha was first recorded at the box default of 20 threads
(`5aff3471…`); the same build at one thread reads `5237c8bd…`, because a float32
reduction changes order with the thread count.
`tests/gate0/test_g03_persist.py` calls `torch.set_num_threads(1)` at **module**
scope, so whether this file's constant matched depended on pytest's collection
order — it passed alone and failed in the full suite. The pin now names one
thread and a fixture sets and restores it. `[MEASURED]` The §4.1 `torch.equal`
proof is unaffected: both sides run in one process at one thread count.

---

## 5. THE PINNING DISCRIMINATOR

**The problem, stated as the test that names it.** Three runs end with
`final β = [1.0, 1.0]` and mean three different things:

```
test_a_frozen_beta_and_a_beta_that_moved_and_returned_have_the_same_final_value
```

is the receipt that the final value cannot tell them apart, and it PASSES — that
is the point. The discriminator is two quantities the final value does not carry:

* **MOBILITY** — did a gradient reach β at every logged step
  (`n_missing_grad == 0` and `max_abs_grad > 0`), and what is the **integrated**
  `|∂L/∂β_i|` over training (`beta_census`);
* **DISPLACEMENT** — did β's value ever leave its init (`max_displacement`), and
  where does it stand now (`final_displacement`).

| `mobile` | `moved` | `state` | reading |
|---|---|---|---|
| False | False | `immobile` | β **could not** move — a bug, not a result. The run answers nothing about pinning. |
| True | False | `not_updated` | β had a live gradient and was never stepped (out of the optimizer). Also void. |
| True | True | `moved` | β moved; `final_displacement` and RULING 2a's criterion say whether it came back. |
| False | True | `moved` | β moved by something other than its gradient. |

### 5.1 The planted negatives, all four measured

| plant | how it is planted | reads |
|---|---|---|
| **β frozen** | `beta.requires_grad_(False)` | `mobile=False`, `moved=False`, `max_abs_grad=0.0`, `census=[0.0, 0.0]`, `requires_grad=False`, `state="immobile"` |
| **β gradient detached** | `CEQAttention._smprime` monkeypatched to pass `self.beta.detach()`; flag still True, parameter still in the optimizer | `mobile=False`, `n_missing_grad = n_logged × n_layers`, `census=[0.0, 0.0]`, `requires_grad=True`, `state="immobile"` |
| **β out of the optimizer** | optimizer built over `named_parameters()` minus `self_attn.beta`; gradient live every step | `mobile=True`, `moved=False`, `max_abs_grad>0`, `max_displacement=0.0`, `state="not_updated"` |
| **the logged column stale** | a cached copy of the first read | live column follows a `fill_(0.25)`, the cache does not |

and the must-fire: an ordinary run reads `mobile=True`, `moved=True`,
`final_displacement > 0`, `state="moved"`.
`test_the_three_states_are_pairwise_distinct` asserts all three void/real states
arise from runs whose final β is `[1.0, 1.0]`. `[MEASURED]`

**The census is what makes this rigorous rather than sampled.** The trajectory is
logged every `log_every`, so `max_displacement` is a **lower bound** — an
excursion entirely between two log points is not seen, and `beta_summary`'s
docstring says so. The census is an **integral over every step**, so a β that
moved and returned between two log points still shows a non-zero census. It is
also the RULING 2a witness that separates PINNED-WITH-SIGNAL from
PINNED-WITHOUT-SIGNAL.

### 5.2 One probe-level fact, reported so nobody reads it as a result

At AdamW `lr = 0.5` on this 27,914-parameter shape, β reads `nan` by step 3.
`[MEASURED]` The probe runs at `lr = 0.02`. This is a fact about the probe's
step size, **not** a training result and not a statement about the arm.

---

## 6. RULING 2a, AND THIS NODE'S ADJUDICATION AGAINST IT

### 6.1 The criterion, as implemented

`β_i` is **PINNED** iff `|β_i,final − 1| ≤ k·δ_β,i` with `k = 5`, `δ_β,i` the
end-of-training spread of that same parameter across the RULING 1 identical-seed
pair. `k` is a named argument (default `5.0`), not a literal buried in a
comparison, and the boundary is tested **on** it and one ulp of β either side.

**Quantifier: per parameter**, and the branches are three.

| branch | condition | implemented as |
|---|---|---|
| A | ≥ 95 % pinned | `frac >= 0.95` |
| B | ≤ 5 % pinned | `frac <= 0.05` |
| C | else — MIXED | `moved_names` lists **which** instances left the corner, by name |

`test_the_three_branches_are_reachable_and_the_thresholds_are_not_a_dichotomy`
checks all three are reachable and that 1-in-20 lands in **B** and not C.

### 6.2 The census word

RULING 2a: the card must use the census's word, not the flattering one. The rule
implemented is the **weaker-by-default** one:

> `word = "preferred"` only if there IS a comparison group (at least one moved β)
> **and** every pinned β's census is at least the **median** census of the moved
> β's. Otherwise `"unused"`.

"Preferred" is a **comparative** claim — softmax was chosen over the alternative
— so it needs the alternative to have been priced. With no moved β there is no
comparison group at all, and the word stays `"unused"` — which is branch A's own
sentence, so **this rule can never upgrade a card sentence and can only refuse
to**. Per-parameter `signal = census_i > 0` is emitted beside it so a single
never-exercised dial is visible.

### 6.3 A degeneracy in 2a, reported rather than papered over

If the identical-seed pair is bitwise on its path then `δ_β,i = 0`, the tolerance
is `0`, and PINNED collapses to β being **exactly** 1.0 — so an ordinary run
reads 0 % pinned and lands in **branch B for a reason about the pair's
determinism, not about training**. `beta_summary` emits
`n_degenerate_floor`, the count of zero-floor entries, so the branch cannot be
read without seeing it. RULING 1 puts training inside the CUDA nondeterminism
hole, so a certified-device pair is expected to give `δ_β > 0`; a pair that does
not is a finding about the pair.
`test_a_bitwise_identical_seed_pair_makes_the_criterion_degenerate`. `[MEASURED]`

### 6.4 This node's own proposal, and the argument against it

The proposal put to the coordinator before 2a landed was: **β is pinned iff
substituting β←1 in the finished checkpoint moves the loss by less than RULING
1's floor.** Its basis was that the loss floor cannot be carried into β's units —
the conversion factor is `dL/dβ`, which is small exactly where the question is
asked, so the tolerance diverges and every β would read pinned — whereas moving
the **parameter into loss space** works and needs no new threshold.

**RULING 2a dissolves that basis** by measuring each quantity's floor in its own
units from the same pair, so the unit objection no longer stands.

**And by "which claims less", 2a wins.** The two criteria are not nested. A β
sitting visibly far from 1 in a layer of low sensitivity **passes** the
substitution test and **fails** 2a — and in that case the substitution criterion
would license the card's strong sentence, "the shipped model **is** the
softmax-corner object", about a model whose β is not 1. That is a stronger claim
than 2a licenses, from a weaker fact. **That is the argument against this node's
proposal and it is made here rather than left to be found.**

**What survives.** `beta_substitution` is kept as a **diagnostic with a
one-directional role**: it may CONTRADICT branch A (if β is pinned but swapping
β←1 moves the loss by more than the floor, the sentence "the shipped model is the
softmax-corner object" is measurably false and the card must not say it), and it
may never ESTABLISH branch A. It is also blind to whether β could move at all —
a frozen model passes it trivially with `delta = 0.0`, which
`test_the_substitution_is_blind_to_whether_beta_could_move` demonstrates and
which is why admissibility (§5) gates everything.

Measured on the §7 demo run: `loss_trained = 1.4357115030288696`,
`loss_at_beta_one = 1.4355229139328003`, `delta = 1.8859e-04`. `[MEASURED]` —
and that is a plumbing number from a 6-step probe, not a reading of anything.

---

## 7. THE EXACT SHAPE OF THE LOGGED COLUMN

One entry of the run record's `beta` list, from a 6-step probe logging every 2
steps plus a final unconditional log (`[MEASURED]`):

```json
{
  "step": 6,
  "name": ["model.layers.0.self_attn.beta", "model.layers.1.self_attn.beta"],
  "beta": [1.034193754196167, 0.9768826365470886],
  "grad": [0.00018533470574766397, 0.0013717601541429758],
  "requires_grad": [true, true]
}
```

and the census, one float per β, integrated over **every** step:

```json
"beta_census": [0.02425182834849693, 0.008847580436849967]
```

`grad` is `null`, not `0.0`, when no backward has run or the parameter is out of
the graph — those are not a zero gradient and recording them as one would erase
the distinction §5 exists for. Everything is JSON-native; the column survives
`json.dumps`/`loads` unchanged, asserted.

**`beta_summary` on that series**, with `delta_beta` supplied
(**`[1e-3, 1e-3]` is a `[ASSUMED]` PLACEHOLDER FOR SHAPE ONLY — it is not a
floor and the branch below is not a reading**):

```json
{"name": [...], "n_layers": 2, "n_logged": 4, "init": 1.0,
 "final": [1.034193754196167, 0.9768826365470886],
 "final_min": 0.9768826365470886, "final_median": 1.0055381953716278,
 "final_max": 1.034193754196167,
 "displacement": [0.03419375419616699, 0.023117363452911377],
 "final_displacement": 0.03419375419616699,
 "max_displacement": 0.03419375419616699, "max_abs_grad": 0.012335949577391148,
 "n_missing_grad": 0, "requires_grad": true, "mobile": true, "moved": true,
 "state": "moved",
 "delta_beta": [0.001, 0.001], "k": 5.0, "pinned": [false, false],
 "pinned_fraction": 0.0, "n_degenerate_floor": 0, "branch": "B",
 "census": [0.02425182834849693, 0.008847580436849967],
 "signal": [true, true],
 "census_median_pinned": null, "census_median_moved": 0.016549704392673448,
 "word": "unused",
 "moved_names": ["model.layers.0.self_attn.beta", "model.layers.1.self_attn.beta"]}
```

Without `delta_beta` the same call returns `"pinned": null, "branch": null` and a
`branch_reason` naming `scripts/k_noise_floor.py`. It does **not** default to a
branch.

---

## 8. WHAT THIS NODE NEEDS FROM OTHER FILES

### 8.1 `ceq/hf/train.py` — five lines, not this node's file

Held by `test_the_run_record_carries_the_beta_column`, a **strict** xfail.

```diff
-from .modeling_ceq import CEQForCausalLM
+from .modeling_ceq import CEQForCausalLM, beta_census, beta_column
@@
-    losses, gnorms, l1mins, slot = [], [], [], 0
+    losses, gnorms, l1mins, slot = [], [], [], 0
+    betas, census = [], []
@@
         loss.backward()
+        beta_census(model, census)   # BEFORE clip_grad_norm_: the ruling's dL/dbeta
         g = torch.norm(...)
@@
         if log_every and step % log_every == 0:
+            betas.append(dict(step=start_step + step, **beta_column(model)))
             print(...)
@@
     for h in handles:
         h.remove()
+    betas.append(dict(step=start_step + steps, **beta_column(model)))
@@
-    record = dict(steps=steps, start_step=start_step,
+    record = dict(steps=steps, start_step=start_step, beta=betas,
+                  beta_census=census,
                   operator=model.config.operator, losses=losses, ...)
```

Three properties of the placement, each load-bearing:

1. **`beta_census` goes immediately after `loss.backward()` and BEFORE
   `clip_grad_norm_`**, so the integral is `|∂L/∂β|` and not the clipped
   gradient. The column, logged after `opt.step()`, carries the **post-clip**
   gradient — the two differ and the report says which is which.
2. **The final append is unconditional and outside the loop**, so the final
   distribution is recoverable whatever `log_every` is and whether or not
   `steps-1` is a multiple of it. `.grad` still holds the last step's gradient
   there; nothing clears it.
3. On `sgate`/`signed` both are empty lists and cost nothing — no guard needed.

### 8.2 `ceq/hf/train.py::_OPERATOR_KEYS` — one line, a real defect

Held by `test_build_forwards_the_smprime_switches`, also a strict xfail.

```python
_OPERATOR_KEYS = ("operator", "rho", "lam", "hops",
                  "smp_beta", "smp_qk", "smp_g")
```

Today `build()` filters `**overrides` through `_OPERATOR_KEYS`, so
`train(operator="smprime", smp_beta=0.0)` **silently drops** `smp_beta` and
trains at `SMPRIME_CORNER` while the caller believes it asked for the (L)
corner. Nothing raises. It is the same defect shape as the missing `operator`
passthrough `tests/gate0/test_g12_train_operator.py` records, one level down.
**It does not block Q3** — Q3 wants the default, β init 1 — which is why it is
filed rather than worked around. `[MEASURED]`, read off the live file.

### 8.3 `scripts/k_noise_floor.py` — nothing new is needed

The floor node's `per_parameter_deltas(out_dir_a, out_dir_b, names="self_attn.beta")`
already emits exactly what RULING 2a's criterion consumes, keyed by
`named_parameters()` name:

```json
"per_parameter_delta": {
  "model.layers.0.self_attn.beta": {"shape": [], "delta": 4.6e-07,
                                    "max_abs_delta": 4.6e-07,
                                    "mean_abs_delta": 4.6e-07}, ...}
```

**The join, one line, by name and never by position:**

```python
delta_beta = [floor["per_parameter_delta"][n]["max_abs_delta"]
              for n in summary["name"]]
```

β is `shape []`, so `delta`, `max_abs_delta` and `mean_abs_delta` coincide;
`max_abs_delta` is named because it is the one that stays correct if β ever gains
an axis. `test_the_column_name_is_the_key_the_floor_node_will_hand_back` pins the
key so the two nodes cannot drift apart silently — a mismatched key would join to
nothing and report every β unpinnable, with nothing raised.

**Ask to the floor node: keep `names="self_attn.beta"` in the call, and keep the
`per_parameter_delta` keys as `named_parameters()` names.** No shape change is
requested.

---

## 9. WHAT THE CARD NODE NEEDS FROM A FINISHED RUN

Everything below is read off `run_record.json` plus one floor record. **No
re-run, no checkpoint reload except for §6.4's optional diagnostic.**

```python
from ceq.hf.modeling_ceq import beta_summary
delta_beta = [floor["per_parameter_delta"][n]["max_abs_delta"]
              for n in beta_summary(rec["beta"])["name"]]
s = beta_summary(rec["beta"], census=rec["beta_census"], delta_beta=delta_beta)
```

1. **Admissibility first.** If `s["state"] != "moved"` the run is **VOID for this
   question** and neither branch may be written: `immobile` means β could not
   move, `not_updated` means it was never stepped. The card must say the run
   failed to exercise the dial, not that β pinned.
2. **`s["branch"]`** — `A`, `B` or `C` — and `s["pinned_fraction"]`.
3. **`s["final"]`, `s["final_min"]`, `s["final_median"]`, `s["final_max"]`** —
   the distribution RULING 2 requires printed.
4. **`s["moved_names"]`** — branch C's content: **which layers** left the corner.
   There is no head axis; see §3.
5. **`s["word"]`** — `"preferred"` or `"unused"`. Use it verbatim; it is set to
   the weaker one whenever the comparison group is absent.
6. **`s["n_degenerate_floor"]`** — if non-zero, the branch is about the pair's
   determinism and must be reported as such (§6.3).
7. **`s["k"] = 5` is a HEURISTIC SCALE, not a CI** — the card says so, per 2a.

The identity clause is unaffected by all of the above: it cites the β = 0
certificate, which §4 shows is bit-identical before and after this node, and
`ceq/arm_smprime.py` was not edited.

---

## 10. LIMITS

`max_displacement` is over the LOGGED steps only, so an excursion entirely
between two log points is invisible to it; the census sees it, and that is why
both are recorded. The census is a sum of per-step magnitudes and therefore
depends on `steps` and on the learning rate — it is comparable **within** one run
(pinned β's against moved β's, which is what 2a asks) and not across runs of
different length. The gradient in the logged column is post-clip while the census
is pre-clip. Every number in this file is CPU, float32, at a 27,914-parameter
shape, from probes of 1–6 steps; none of them is a training result, a device
timing, or a reading of the arm. `δ_β` does not exist yet, so no branch has been
taken by anything here and the placeholder in §7 is `[ASSUMED]`. The four
`tests/chase` failures in §j are peer-owned and pre-existing by name; this node
did not run a controlled before/after on them and cites
`V17_ARM_WIRING.md` §9 for the identical baseline `[INHERITED]`.
