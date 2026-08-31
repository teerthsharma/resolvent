# V17-K — WIRING `ceq/arm_smprime.py` INTO THE HF LM (Q3's BLOCKER)

**LEAD CAVEAT, BEFORE ANY NUMBER BELOW.** This is PLUMBING and nothing else.
Nothing here was trained as a research reading: the 40 gradient steps report
whether a gradient is finite and report nothing else — no loss, no NRMSE, no
cell, no bed, no seed pool, no verdict. **No number in this file says the arm is
good, bad, or comparable to anything.** It says the arm can now be selected,
built, forwarded, checkpointed and stepped from `CEQConfig`, that the existing
default path is bit-identical to what it was, and what the arm costs in
parameters against a matched control. Whether the arm should be trained at that
corner, and what the comparison means, is the author's ruling.

**SECOND CAVEAT: THE PARAMETER COUNTS ARE NOT EQUAL.** They differ by
`n_layers * (2*(d+1) + 3)` — the arm's own two per-position heads and three
switches — and that difference is *reported*, not fudged. See §5.

**THIRD CAVEAT: THIS IS NOT THE CORNER THE ARM'S BIND IS CLAIMED AT.** The
module docstring claims `beta = 0`, QK-OFF. The LM runs at `beta = 1, qk = 1,
g = 1`, which is `ArmSMPrime.__init__`'s own switch initialisation, because
`qk = 0` deletes the content term exactly and an LM at that corner has no
query-key channel at all. The corner is NAMED IN CONFIG (`SMPRIME_CORNER`), and
§3 states exactly what does and does not carry over.

**Files written.** `ceq/hf/modeling_ceq.py`, `ceq/hf/configuration_ceq.py`
(both edited), `tests/gate0/test_g10_arm_wiring.py` (new), this file. **Nothing
else was touched** — in particular not `ceq/hf/train.py`, not
`ceq/arm_smprime.py`, not `ceq/arm_phase.py`, not `scripts/`, not `scale/`. **No
git command that writes was run.**

**Provenance.** `git rev-parse HEAD` = `ab5b48547884e04258276e6e808d5a71ea65f917`
at start **and** at end (no commit was made by this node). `git status
--porcelain` at start: `?? ceq/kdata.py`, `?? tests/gate0/`. At end: seven
peer-owned paths had appeared and `ceq/arm_smprime.py`, `ceq/hf/train.py` had
been modified **by other nodes** — the `arm_smprime` diff is a `label_cell` BOS
repair (`V17_LABEL_CELL_REPAIR.md`), untouched by this node and unread by this
wiring. Box: Windows 11, RTX 4060 Laptop 8 GiB, torch `2.5.1+cu121`,
transformers `5.3.0`, python 3.11. **Every measurement below is CPU, float32,
at a tiny shape, made this run.** L-TIME: no timing is inherited and none is
reported — device timing belongs to another node.

---

## VERDICT TABLE

| question | answer |
|---|---|
| **(a) Is the arm selectable as the LM's attention operator, from config alone?** | **YES.** `CEQConfig(operator="smprime")` builds `CEQForCausalLM` and forwards. `operator` is already the field `ceq/hf/train.py::build()` splats into `CEQConfig`, so the seam was where the task said it was. `[MEASURED]` |
| **(b) Is the existing default path bitwise unchanged?** | **YES, bitwise.** Weights and logits `torch.equal` against `ceq/hf/{modeling,configuration}_ceq.py` as they stood at commit `ab5b485`, rebuilt into a throwaway package and run side by side at the same seed. Not `allclose`. `[MEASURED]` |
| **(c) The two parameter counts** | **NOT EQUAL, and the excess is named.** At `d=32, L=2, H=4, V=32, S=16`: arm **27,914**, softmax-shaped control **27,776**, difference **138** `= L * (2*(d+1) + 3)`. At `ceq/hf/train.py::DEFAULTS` (`d=512, L=8, H=8, V=256, seq=512`): arm **25,736,232**, control **25,728,000**, difference **8,232** = **+0.03200 %**. Cross-checks `V16_ARM_SMPRIME.md` §1's `4,806 − 4,769 = 37 = 2·16 + 5` at `d_model = 16`. `[MEASURED]` |
| **(d) 40 gradient steps from the initialiser, finite?** | **YES.** `first_non_finite = None` at 40 steps, CPU, float32, `operator="smprime"`, AdamW `lr=1e-3`, batch 2 × seq 8. Reports finiteness and nothing else (L-LEAN). `[MEASURED]` |
| **(e) Did anything require a NEW CONSTRUCTION?** | **THREE PLACES WANTED ONE. NONE WAS BUILT.** (1) a padding-mask path through the arm — the seam **raises** instead; (2) a Hub-portable duplicate of the arm's operator — the seam **imports the arm** instead; (3) a per-head widening of the gate heads — the seam uses the arm's own `d → 1` shape instead. §7. |
| **(f) Did anything from `ceq/hf/train.py` turn out to be needed?** | **`build()` needed nothing — but `train()` cannot reach it.** `train()` calls `build(hidden_size=…, n_layers=…, n_heads=…, seq=…, vocab_size=…)` and forwards no operator key and accepts none, so **Q3 would train `sgate` whatever the config said**. That is a three-line passthrough in a file this node may not edit. §8. `[MEASURED]` — read off the live file at end of run. |
| **(g) Is every check non-degenerate?** | **YES, each with a planted perturbation that fires.** Unknown operator name still refused; a magnitude-head bias bump moves the logits; one float32 ulp on `lam` and one ulp on `lm_head.weight` break the bitwise comparison; a doubled hidden size breaks the parameter accounting; a NaN planted at step 3 is caught at step 3; rewriting the saved `config.json`'s operator changes what the reloaded checkpoint computes. §6. |
| **(h) Does a checkpoint round-trip?** | **YES, bitwise.** `save_pretrained` → `from_pretrained` reproduces the logits under `torch.equal`, with `operator="smprime"`, `(smp_beta, smp_qk, smp_g) = (1.0, 1.0, 1.0)` and `(rho, lam, hops) = (None, None, None)` preserved. Q3's deliverable is a checkpoint, so this is part of the seam. `[MEASURED]` |
| **(i) Regression** | `tests/arm_smprime` **52 passed** before and **52 passed** after. The four HF-package test files in `tests/chase` read **4 failed / 119 passed / 2 xfailed** with the pre-change files restored **and** with the wiring — **the same four by name**, both about the WITHDRAWN decay exponent and neither about this change. §9. |
| **(j) GREEN/RED/BLOCKED** | **GREEN on the wiring.** `tests/gate0/test_g10_arm_wiring.py` **11 passed** from **8 failed / 2 passed** RED. Q3 is unblocked on the model side and **still blocked on `train()`'s passthrough**, which is another node's file. |

---

## 0. RED

Tests were written first, against a tree where `operator="smprime"` did not
exist. `python -m pytest tests/gate0/test_g10_arm_wiring.py -q`:

```
E   ValueError: unknown operator 'smprime'; expected one of ('sgate', 'signed').
    The shipped default is 'sgate', the operator the 1.0334 parity median was
    measured on. 'signed' is the L1-normalized negative control and measured 1.337x.

ceq\hf\configuration_ceq.py:99: ValueError

FAILED tests/gate0/test_g10_arm_wiring.py::test_the_arm_is_selectable_from_config_alone
FAILED tests/gate0/test_g10_arm_wiring.py::test_the_gate_heads_reach_the_operator
FAILED tests/gate0/test_g10_arm_wiring.py::test_the_arm_refuses_a_padding_mask_instead_of_ignoring_it
FAILED tests/gate0/test_g10_arm_wiring.py::test_a_one_ulp_weight_change_is_caught_by_the_bitwise_comparison
FAILED tests/gate0/test_g10_arm_wiring.py::test_the_arm_and_the_softmax_shaped_control_differ_by_named_parameters
FAILED tests/gate0/test_g10_arm_wiring.py::test_a_mismatched_hidden_size_is_caught_by_the_param_accounting
FAILED tests/gate0/test_g10_arm_wiring.py::test_forty_steps_on_the_arm_have_no_non_finite_gradient
FAILED tests/gate0/test_g10_arm_wiring.py::test_a_planted_nan_is_caught_by_the_gradient_check
8 failed, 2 passed, 2 warnings in 11.81s
```

**The two that passed at RED are the two that were supposed to.**
`test_an_unknown_operator_is_still_refused` exercises a guard that already
existed, and `test_the_default_path_is_bitwise_the_pre_change_code` compares the
tree against itself before the edit — a regression guard passes before the
regression. That is why its planted negative is a separate test, and that test
was RED.

**ONE TEST WENT RED AS WRITTEN AND STAYED RED FOR A REAL REASON, AND THE CLAIM
WAS WEAKENED TO WHAT IS TRUE.** As first written, the plant for the bitwise
proof was one ulp on `model.layers.0.self_attn.qkv.weight[0, 0]`, which the task
brief names. It **does not move the fp32 logits at all** — measured, and not at
64 ulps either:

```
model.embed_tokens.weight                     caught at   1 ulp  max|d|=5.960e-08
model.layers.0.self_attn.qkv.weight           NOT caught up to 64 ulp
model.layers.0.self_attn.o_proj.weight        caught at   1 ulp  max|d|=5.960e-08
lm_head.weight                                caught at   1 ulp  max|d|=1.490e-08
model.layers.1.self_attn.qkv.weight           caught at  16 ulp  max|d|=2.980e-08
model.layers.0.mlp.0.weight                   caught at  16 ulp  max|d|=5.960e-08
```

The `sgate` operator renormalizes, so a sub-ulp logit perturbation in the first
layer is rounded away before it reaches the read-out. The plant was moved to
where it is **on target and visible**: one float32 ulp on `lam`, a knob read by
`sgate_operator` and by nothing else (`max |delta| = 5.960e-08`, caught), plus
one ulp on `lm_head.weight` (`1.490e-08`, caught). §4 states the resulting limit
on what the bitwise proof covers.

## 0b. GREEN

```
$ python -m pytest tests/gate0/test_g10_arm_wiring.py -q -s
  arm  (operator=smprime) parameters = 27,914
  ctrl (operator=sgate)   parameters = 27,776
  difference = 138  = n_layers * (2*(d+1) + 3) = 138
  40-step gradient probe  cpu  float32  operator=smprime  -> first non-finite: None
11 passed in 6.44s
```

---

## 1. THE GAP, STATED PRECISELY

`CEQ_V16_CONTRACT.md` fixes Q3's workhorse as `ceq/arm_smprime.py`. The training
path is `ceq/hf/train.py::train()` → `build()` → `CEQForCausalLM` →
`CEQBlock` → `CEQAttention`, and `CEQAttention` dispatched on `config.operator`
over exactly `{"sgate", "signed"}`. **Nothing in that chain named the arm.** The
arm was reachable only from `scale/m3_capability.py`'s harness and from
`scripts/v15_r1.py`, neither of which produces a HuggingFace checkpoint.

The seam is `config.operator`, exactly as briefed: `build()` already collects
`_OPERATOR_KEYS` from its `**overrides` and splats them into `CEQConfig`, so a
new value of an existing field is the whole change on that side.

## 2. WHAT WAS WIRED, AND WHAT WAS NOT

**What is called is the arm's OPERATOR, not the `ArmSMPrime` container.** The
class carries `wq`, `wk`, an MLP, a scalar `readout` and a `forward` that
returns `[n]` by reading position `s − 1` — that is `scale/m3_capability.py`'s
regression harness, not an attention block. Embedding it whole would have put a
`d→128→d` MLP and a `d→1` read-out into every layer, dead, and would have broken
the very parameter comparison Q3 needs. So `CEQAttention` at `operator="smprime"`
calls **`ceq.arm_smprime.readout`** — which is `operator` → `numerator` → `hop` →
`path_product` → `gate` → `blend`, the shipped module, unmodified — and carries
**the arm's own parametrization**: `m_head`, `theta_head` (`nn.Linear(d, 1)`
each) and `beta`, `qk`, `g` as `nn.Parameter`s, which is `ArmSMPrime.__init__`
minus the harness pieces. The `4,806 − 4,769 = 37 = 2·16 + 5` excess
`V16_ARM_SMPRIME.md` §1 records at `d_model = 16` is the arithmetic check that
this is the same parametrization and not a nearby one.

**The gate heads are shared across attention heads.** `m_head` is `d → 1`,
giving `[B, S]`, unsqueezed to `[B, 1, S]` so the resulting `[B, 1, S, S]` path
product broadcasts over the operator's `[B, H, S, S]`. A per-head widening to
`nn.Linear(d, n_heads)` would be a construction the arm does not have. Not
built.

**No path sum.** The arm's read-out is `O_i = Σ_j W_ij V_j`, one application.
`path_sum` is not called at this operator and `hops` has no meaning there — see
§5 on why `CEQConfig` refuses to carry one rather than storing an unread number.

**Causal inclusive of the diagonal.** `ceq_operator` and `sgate_operator` are
STRICTLY causal (`tril(-1)`) because the path sum needs `A` nilpotent. The arm's
window is `Ico (j+1) (i+1)`, empty at `j = i`, so `G_ii = 1` and the operator is
causal over `j ≤ i`. Position `i` reading its own value is ordinary causal
attention and not label leakage — the LM predicts token `i+1` from position `i`,
and `v_i` is a projection of token `i`. **Stated because the two operators in the
same file differ on it**, not because it is a defect.

## 3. THE CORNER, NAMED IN CONFIG

`ceq/arm_smprime.py`'s docstring claims the (L) bind at **`beta = 0`, QK-OFF**,
read-out `O_i = Σ_j G_ij V_j` with `V = b` **unrescaled** — which is why neither
`log m` nor `1/(1−m)` is ever instantiated and why both endpoints of the closed
`[0,1]` magnitude interval are ordinary points.

The LM does **not** run there, and the difference is now a named constant,
`configuration_ceq.SMPRIME_CORNER = (1.0, 1.0, 1.0)`, with `smp_beta`, `smp_qk`,
`smp_g` as config fields.

| | (L) bind corner | LM corner as wired |
|---|---|---|
| `beta` | `0` | `1.0` |
| `qk` | OFF (`0`) | `1.0` |
| `g` | `1` | `1.0` |
| route | `"product"` | `"product"` |
| `V` | `b`, unrescaled | `v` from `qkv`, unrescaled |

**Why not the bind corner.** `qk = 0` deletes the content term *exactly*
(`0.0 * w = 0.0`, `exp(0) = 1`), so an LM there has no query-key channel at all
and its `q`/`k` projections receive no gradient. `qk = 1` is forced by the task,
not chosen for taste.

**`(1, 1, 1)` IS NOT A CORNER THIS NODE INVENTED.** It is
`ArmSMPrime.__init__`'s own initialisation, and the module states why: "`beta =
1` is the only initialisation inside the softmax class (`#5b`), which is where an
arm built FROM softmax should start." All three are `nn.Parameter`s on the block
exactly as on the arm, so this names where training **starts**, not where it
stays.

**WHAT DOES NOT CARRY OVER, AND THIS IS THE HONEST COST OF THE MOVE.**
`V16_ARM_SMPRIME.md` (h) reads: at `beta = 1` the label bind **fails at
1.335288** and no finite compensation exists at `m = 1`. So the identity result
the arm is certified by — residual `5.92e-16` on BED-M's real support — is a
`beta = 0` result and **says nothing about the LM at `beta = 1`**. What survives
the move is structural and not numerical: the path product still annihilates
exactly at a zero magnitude (clause 4), `Z_i > 0` still holds because the
diagonal term is the empty product, and no logarithm of the magnitude is taken
at any `beta`. **`beta` is trainable, so the LM may leave `beta = 1`
immediately; nothing here measures whether it does.**

**Measured at the initialiser, and it is worth the author's eye.** With
transformers' `_init_weights` (`normal_(std=0.02)`, bias zeroed) the magnitude
head starts near zero, so the closed lower cap binds on most positions:

```
layer 0  frac m == 0 = 0.641602   (n = 1024 positions)   max m = 2.084891e-01
layer 1  frac m == 0 = 0.514648   (n = 1024 positions)   max m = 3.494641e-01
```

That is the arm's own `identity_heads()` warning (`V15_ARM_PHASE.md` §7 item 4)
arriving through a different door: a head initialised at zero puts a closed-cap
arm at the **annihilating** gate, not the identity. `ArmSMPrime.identity_heads()`
exists to move it to `m = 1`; the LM does not call it, because calling it would
be choosing an initialiser, which is a construction and the author's ruling.
**Reported, not fixed.**

## 4. THE EXISTING PATH IS BITWISE UNCHANGED

`tests/gate0/test_g10_arm_wiring.py::test_the_default_path_is_bitwise_the_pre_change_code`
writes `ceq/hf/configuration_ceq.py` and `ceq/hf/modeling_ceq.py` **as they stood
at commit `ab5b48547884e04258276e6e808d5a71ea65f917`** into a throwaway package,
imports it, builds `CEQForCausalLM(CEQConfig(**TINY))` under the same
`torch.manual_seed(0)` in both, and compares:

* every tensor of `state_dict()` under `torch.equal`, key sets equal;
* the forward logits under `torch.equal`.

Both pass. Not `allclose`: a plumbing change that moves the default path by
`1e-16` is not plumbing.

**The reference is pinned to a literal commit sha, not to `HEAD`.** A test that
compares the working tree against `HEAD` passes trivially the moment the change
is committed — that would have been vacuous control number fifteen.

**LIMIT ON WHAT THIS PROVES.** It proves the default path is bit-identical at
this shape and this seed. It does **not** prove that every sub-ulp perturbation
anywhere in the model would have been visible: measured on this box, `torch.equal`
on fp32 logits does not see one ulp — nor 64 ulps — on
`model.layers.0.self_attn.qkv.weight[0, 0]`. It does see one ulp on `lam`, on
`lm_head.weight`, on `embed_tokens.weight` and on `o_proj.weight`, and 16 ulps on
`layers.1.self_attn.qkv.weight`. The table is in §0. The plant that guards the
claim is therefore placed **inside the operator** (`lam`), which is the thing the
claim is about.

**Structurally, the default path cannot have moved**: the branch added to
`CEQAttention.forward` is `if self.operator == "smprime": … else: <the previous
body, verbatim>`, and the extra parameters are constructed only inside
`if self.operator == "smprime":`, so the module-construction order — and hence
every draw from the RNG — is unchanged at `sgate` and `signed`. The bitwise test
is the receipt; this paragraph is not.

## 5. MATCHED PARAMETERS — THE COUNTS, AND THE DIFFERENCE

The control is the same LM at the same shape on the existing operator.
`CEQAttention` at `sgate` carries the qkv and output projections a softmax block
carries and the operator has no parameters of its own, which is what makes the
comparison possible at all (`ceq/hf/modeling_ceq.py`, and `COSTS["parity"]`
records `3,319,296 parameters exactly on both arms` for that reason).

| shape | `operator="smprime"` | `operator="sgate"` (softmax-shaped control) | difference |
|---|---|---|---|
| `d=32, L=2, H=4, V=32, S=16` | **27,914** | **27,776** | **+138** |
| `ceq/hf/train.py::DEFAULTS`: `d=512, L=8, H=8, V=256, seq=512` | **25,736,232** | **25,728,000** | **+8,232** (+0.03200 %) |

`138 = 2 · (2·(32+1) + 3)`; `8,232 = 8 · (2·(512+1) + 3)`. **THEY ARE NOT EQUAL
AND THE TEST DOES NOT PRETEND THEY ARE.** The assertion is that the excess is
`n_layers · (2·(d+1) + 3)` **and that it is exactly these tensors, by name**:

```
model.layers.{i}.self_attn.m_head.weight       [1, d]
model.layers.{i}.self_attn.m_head.bias         [1]
model.layers.{i}.self_attn.theta_head.weight   [1, d]
model.layers.{i}.self_attn.theta_head.bias     [1]
model.layers.{i}.self_attn.beta                []
model.layers.{i}.self_attn.qk                  []
model.layers.{i}.self_attn.g                   []
```

and that the control has **no** parameter the arm lacks. This is the arm's own
parametrization — two per-position scalar heads and three scalar switches — and
it is the same excess `V16_ARM_SMPRIME.md` §1 already recorded against a softmax
control: `4,806 − 4,769 = 37`, and `37 = 2·16 + 5` at `d_model = 16`.

**FINDING FOR THE AUTHOR.** A Q3 comparison "at matched params" is therefore
matched to **+0.032 %** at `train.py`'s shape and not exactly. Three ways to make
it exact all cost something and none was taken: shrinking the arm's `d` breaks
"same shape"; widening the control's `d` is not expressible (the excess is not a
multiple of any control parameter block); deleting a head or a switch mutilates
the arm. **The difference is reported so the author can rule.** It is also worth
noting which way it cuts: the arm carries **more** parameters, so a favourable
reading for the arm would be the one needing the caveat.

**`rho`, `lam` and `hops` are refused at `smprime`, not defaulted.**
`OPERATOR_DEFAULTS["smprime"] = (None, None, None)`, and passing any of them
raises. The arm does not normalize by a row L1 (no `rho`), has no negative
softmax half (no `lam`) and takes no path sum (no `hops`). Storing a number in
any of those slots would produce a `config.json` that reads like a configured
operating point and that no line of code loads — the same defect the file's first
guard was written for.

## 6. NON-DEGENERACY — EVERY PASS HAS A PLANT THAT FIRES

This repository has struck **14 vacuous controls across five authors**, so each
claim carries a perturbation of the thing it is supposed to notice.

| claim | plant | result |
|---|---|---|
| selectable from config | `operator="smprme"` (typo) | `ValueError: unknown operator` — the third name did not turn validation into a pass-through `[MEASURED]` |
| the gate reaches the operator | `+0.5` on `layers[0].self_attn.m_head.bias` | logits move. **This is the arm's own historical failure mode**: `V16_ARM_SMPRIME.md` §0 records `label_cell` computing a `route` and never passing it, so the mutilation was silently discarded. A head computed and dropped would have given a gate-free operator wearing the arm's name `[MEASURED]` |
| the padding mask is not silently dropped | pass `attention_mask` at `smprime` | `NotImplementedError` raised `[MEASURED]` |
| default path bitwise | 1 float32 ulp on `lam` (read by `sgate_operator` and nothing else) | `torch.equal` breaks, `max\|delta\| = 5.960e-08` `[MEASURED]` |
| default path bitwise | 1 ulp on `lm_head.weight[0,0]` | `torch.equal` breaks, `max\|delta\| = 1.490e-08` `[MEASURED]` |
| parameter accounting | control built at `2d` (mismatched hidden size) | the accounting no longer holds `[MEASURED]` |
| gradient finiteness | NaN written into `layers[0].self_attn.qkv.weight[0,0]` at step 3 | caught at step 3, exactly `[MEASURED]` |
| the checkpoint carries the operator | rewrite the saved `config.json` to `operator="sgate"` | the reloaded model computes something else — the operator is read from config, not baked into weights `[MEASURED]` |

One more separation, not a plant but a check that the branch is not a no-op:
at identical seed and identical shape, `operator="smprime"` and
`operator="sgate"` differ by `max |delta| = 4.871254e-01` on the logits.
`[MEASURED]`

## 7. WHERE A NEW CONSTRUCTION WOULD HAVE BEEN NEEDED — AND WHAT WAS DONE INSTEAD

The round's first law strikes new constructions. Three came up.

**(1) A padding-mask path through the arm.** `ceq/arm_smprime.py::operator` takes
no `attention_mask`. `sgate_operator` masks *before* its softmax precisely
because zeroing afterwards would leave masked keys holding mass — and this
project has twice shipped a constraint reported applied and not applied (a
`BlockMask` that `to_dense()` showed sparse while the kernel ran dense; an
`is_causal` correct in prefill and wrong at decode). Threading a mask into the
path product is a change **to the arm**, a module this node may not edit.
**NOT BUILT. The seam raises `NotImplementedError`** with the reason. Causality
is unaffected — the path product is masked to `j ≤ i` unconditionally, so a
caller who omits a mask loses padding, never causality. `train.py` calls
`model(input_ids=x, labels=x)` with no mask, so **Q3 is not affected**.

**(2) A Hub-portable duplicate of the arm's operator.** `modeling_ceq.py`
duplicates `ceq/attention.py` and `ceq/lm.py` because the Hub copies the modeling
file into a flat repository where `from ceq… import` cannot resolve; each
duplicate carries a bitwise oracle test against its source. A third duplicate
would be a **second implementation** of the module the contract fixes as the
workhorse ("Workhorse arm_smprime only"), and Q3 would then train the copy rather
than the arm. **NOT BUILT. The arm is imported instead, inside the branch**, so:

* `sgate` and `signed` are untouched and the Hub copy still **loads** and **runs**
  on a machine without this repository — the same argument the module docstring
  makes about `import triton` at module scope;
* a Hub download that selects `operator="smprime"` raises `ImportError` — loudly,
  at the branch, not silently.

**This is a real cost and it is the author's to rule on.** Making `smprime`
Hub-portable requires duplicating the arm; that is a construction, so it is
reported rather than built.

**(3) A per-head gate.** The arm's heads are `d_model → 1`: one magnitude and one
phase per **position**. A multi-head LM invites `nn.Linear(d, n_heads)`.
**NOT BUILT** — the heads are shared across attention heads and the gate
broadcasts, which is the arm's shape verbatim and also the smaller parameter
excess.

Two further things were deliberately **not** repaired, both being the arm's own
numerics: there is no max-subtraction before `exp` in `numerator` (so large
logits overflow, as they do in the arm), and the initialiser leaves 51–64 % of
positions at `m = 0` (§3). Repairing either is a new construction.

## 8. WHAT `ceq/hf/train.py` STILL NEEDS — REPORTED, NOT EDITED

`build()` needed **no** change: it already collects the operator keys from
`**overrides` and splats them into `CEQConfig`, which is why the seam is a config
field.

**But `train()` never passes them.** Read off the live file at the end of this
run (that file is being edited by another node; this is what it says now):

```python
def train(*, out_dir, steps, batch, seq, hidden_size, n_layers, n_heads,
          device="cuda", vocab_size=256, …):
    …
        model = build(hidden_size=hidden_size, n_layers=n_layers, n_heads=n_heads,
                      seq=seq, vocab_size=vocab_size).to(device)
```

`train()` accepts no operator keyword and forwards none, so **every `train()` run
builds `sgate` regardless of what the caller wants** — this is true today for
`signed` as well, so it is a pre-existing gap this node found rather than one the
wiring created. The fix is a passthrough (`**operator` on the signature, splatted
into the `build(...)` call), in a file this node may not edit. **Q3 remains
blocked on exactly that.**

**One note for that file's owner.** `_attach_row_l1_probe` hooks
`layer.self_attn.qkv` and reads `attn.n_heads` / `attn.d_head`, all of which
`CEQAttention` still carries at `smprime`, so the probe **runs**. But the number
it logs is the row L1 of the raw logits, and at `smprime` the operator does not
divide by it at all — the arm's normalizer is `Z_i^beta` over the modulus row.
The probe's docstring already distinguishes `signed` (blow-up alarm) from `sgate`
(logit-spread diagnostic); at `smprime` it is neither, and the docstring does not
yet say so.

## 9. REGRESSION

Counts are stated with `git rev-parse HEAD` and `git status --porcelain` beside
them because another node is independently measuring suite counts and **the tree
moved under this run**: seven peer-owned paths appeared and two peer-owned files
changed between the first and last measurement.

| suite | before | after |
|---|---|---|
| `tests/arm_smprime` | **52 passed** | **52 passed** |
| `tests/gate0` | first attempt **collection ERROR** (`cannot import name 'k_cert' from 'scripts'`); second attempt minutes later **18 failed / 66 passed** | **4 failed / 161 passed** |
| `tests/gate0/test_g10_arm_wiring.py` (this node's) | **8 failed / 2 passed** (RED) | **11 passed** |
| `tests/chase/test_hub_package_hardening.py`, `test_ceq_hub_package.py`, `test_colab_chain.py`, `test_capability_table.py` | **4 failed / 119 passed / 2 xfailed** | **4 failed / 119 passed / 2 xfailed**, same four by name |
| `tests/loop/test_no_struck_constant_ships.py`, `tests/loop/test_measured_operator_is_shipped.py`, `tests/chase/test_module_prose_is_bound.py` | not measured separately | **42 passed** |

`HEAD` = `ab5b48547884e04258276e6e808d5a71ea65f917` at every one of these runs.
`git status --porcelain` at the first `tests/gate0` run: `?? ceq/kdata.py`,
`?? tests/gate0/`. At the last: `M ceq/arm_smprime.py`, `M ceq/hf/configuration_ceq.py`,
`M ceq/hf/modeling_ceq.py`, `M ceq/hf/train.py`, plus eleven untracked peer paths.
**Only the two `ceq/hf/` files are this node's** — the `arm_smprime` and
`train.py` modifications are other nodes'.

**`tests/gate0` IS A MOVING TARGET AND ITS COUNTS ARE NOT A READING OF THIS
CHANGE.** It went from failing to collect, to `18 failed / 66 passed`, to
`4 failed / 161 passed` while peers landed `scripts/k_cert.py`, `ceq/autopilot.py`
and their fixtures. The eleven tests this node added are inside the last figure.
None of the four residual failures is in this node's file.

**THE FOUR `tests/chase` FAILURES ARE PRE-EXISTING AND WERE MEASURED AS SUCH.**
The two `ceq/hf/` files were temporarily replaced with their `ab5b485` content,
the four HF-package test files were run, and the files were restored (verified
byte-identical). The result was the same **4 failed / 119 passed / 2 xfailed**,
the same four names, both of which assert
`COSTS["content_conditional_sign_decay"]["slope"] == -1.389` against the dict's
WITHDRAWN `None`. Untouched by this change.

**Device.** One CUDA check only, at the smallest shape that binds it
(`[1, 4]`, `d=32`, `H=2`, `L=1`): forward and backward both produce finite values
with `operator="smprime"`. `[MEASURED]` No timing was taken — that is another
node's measurement and L-TIME forbids inheriting one.

## 10. WHAT Q3 CAN NOW TRAIN THAT IT COULD NOT BEFORE

**Before:** `ceq/hf/train.py::train()` could build only `CEQAttention` over
`sgate` or `signed`. `ceq/arm_smprime.py::ArmSMPrime` was instantiated by
`scale/m3_capability.py`'s harness and by `scripts/v15_r1.py`, neither of which
produces a HuggingFace checkpoint. There was **no path at all** from the contract's
workhorse to a `save_pretrained` directory.

**After:** `CEQConfig(operator="smprime")` builds a `CEQForCausalLM` whose
attention is `ceq/arm_smprime.py`'s operator — the shipped module, imported, not
copied — carrying the arm's own two heads and three trainable switches. It
forwards, it takes 40 AdamW steps with finite gradients, it `save_pretrained`s
and `from_pretrained`s back bitwise with the operator and its switches recorded
in `config.json`, and the softmax-shaped control is the same model at
`operator="sgate"`, differing by a counted **+0.032 %** in parameters at
`train.py`'s shape.

**What remains blocked, and it is one file this node may not touch:**
`train()` does not forward the operator keyword to `build()`. Until it does,
`train(...)` builds `sgate` whatever the caller asks for.

**What this file does NOT license.** No comparison, no ratio, no parity claim, no
statement that the arm is or is not competitive. The arm's certified identity
result is a `beta = 0` result and the LM starts at `beta = 1`, where
`V16_ARM_SMPRIME.md` (h) records the label bind failing at `1.335288`. Everything
the LM would say about this arm is unmeasured.
