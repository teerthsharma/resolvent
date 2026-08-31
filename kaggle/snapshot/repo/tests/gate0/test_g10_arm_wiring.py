"""G0.10 -- `ceq/arm_smprime.py` AS THE HF LM's ATTENTION OPERATOR, AS PLUMBING.

WHAT WAS MISSING. `CEQ_V16_CONTRACT.md` fixes the workhorse for Q3 as
`ceq/arm_smprime.py` ("Workhorse arm_smprime only; deterministic cumprod; the
complex arm never trains"), and `ceq/hf/train.py::train()` trains
`CEQForCausalLM`, whose attention is `sgate` or `signed` and nothing else.
Nothing in the training path instantiated the arm, so Q3 could not train the
arm the contract names. This suite is the receipt for the seam and for nothing
else -- it is PLUMBING, and a plumbing change that moves either operator is not
plumbing.

FOUR CLAIMS, EACH WITH ITS PLANTED NEGATIVE. This repository has struck
fourteen vacuous controls across five authors, so every assertion below is
followed by a test that perturbs the thing it is supposed to notice and
requires it to fire.

    selectable          the LM builds and forwards from config alone,
                        arm and existing operator both
                        PLANT: an unknown operator name must still be refused,
                        and the gate heads must reach the operator -- the V16
                        report records `label_cell` computing a `route` and
                        never passing it, so a head that is computed and
                        discarded is this arm's own historical failure mode

    bitwise             the DEFAULT path is bitwise what it was before the
                        edit, compared against the modeling and configuration
                        files as they stood at commit `BASE_COMMIT`, rebuilt
                        into a throwaway package and run side by side
                        PLANT: one ulp on one weight must break it

    matched params      the arm and the softmax-shaped control at the same
                        shape, counted and DIFFERENCED, with every excess
                        parameter named
                        PLANT: a mismatched hidden size must break the
                        accounting

    finite gradients    40 steps from the initialiser, CPU, tiny shape
                        PLANT: a planted NaN must be caught

WHY THE REFERENCE IS PINNED TO A COMMIT SHA AND NOT TO `HEAD`. A test that
compares the working tree against `HEAD` passes trivially the moment the change
is committed -- that is vacuous control fifteen. `BASE_COMMIT` is the literal
sha the wiring was written against, so the comparison stays a comparison.

NOTHING HERE TRAINS AS A RESEARCH READING (L-LEAN). The 40 steps report whether
a gradient is finite. No loss, no checkpoint, no cell, no verdict.
"""
from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: The commit the wiring was written against. `ceq/hf/modeling_ceq.py` and
#: `ceq/hf/configuration_ceq.py` are both clean at this sha, so it is the
#: pre-change code by definition and stays so after the change is committed.
BASE_COMMIT = "ab5b48547884e04258276e6e808d5a71ea65f917"

#: Tiny, CPU, and deliberately not a shape anybody trains: the readings here
#: are structural (bitwise equality, parameter counts, finiteness) and none of
#: them depends on the size. Device timing belongs to another node.
TINY = dict(vocab_size=32, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, max_position_embeddings=16)
SEQ, BATCH = 8, 2


# ------------------------------------------------------------------ fixtures

def _cfg(**kw):
    from ceq.hf.configuration_ceq import CEQConfig
    return CEQConfig(**{**TINY, **kw})


def _lm(**kw):
    from ceq.hf.modeling_ceq import CEQForCausalLM
    torch.manual_seed(0)
    return CEQForCausalLM(_cfg(**kw))


def _tokens(seed=1, vocab=32, batch=BATCH, seq=SEQ):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(vocab, (batch, seq), generator=g)


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


@pytest.fixture(scope="module")
def base_pkg(tmp_path_factory):
    """`ceq/hf/` as it stood at `BASE_COMMIT`, importable as a package.

    The modeling file imports its configuration RELATIVELY, so the two blobs
    are written into a real package directory rather than loaded by path.
    """
    d = tmp_path_factory.mktemp("base") / "_ceq_hf_at_base"
    d.mkdir()
    (d / "__init__.py").write_bytes(b"")
    for name in ("configuration_ceq.py", "modeling_ceq.py"):
        blob = subprocess.run(
            ["git", "show", "{}:ceq/hf/{}".format(BASE_COMMIT, name)],
            cwd=str(ROOT), capture_output=True, check=True).stdout
        (d / name).write_bytes(blob)
    sys.path.insert(0, str(d.parent))
    try:
        return importlib.import_module("_ceq_hf_at_base.modeling_ceq")
    finally:
        sys.path.pop(0)


# ------------------------------------------------------------- 1. SELECTABLE

def test_the_arm_is_selectable_from_config_alone():
    """`operator="smprime"` builds and forwards. So does the existing default.

    From CONFIG, which is the seam: `ceq/hf/train.py::build()` already splats an
    `operator` dict into `CEQConfig`, so a config field is all the training path
    needs and `train.py` is not edited by this node.
    """
    x = _tokens()
    for op in (None, "sgate", "signed", "smprime"):
        model = _lm() if op is None else _lm(operator=op)
        out = model(input_ids=x).logits
        assert out.shape == (BATCH, SEQ, TINY["vocab_size"]), (op, out.shape)
        assert torch.isfinite(out).all(), op
    assert _lm(operator="smprime").config.operator == "smprime"


def test_an_unknown_operator_is_still_refused():
    """PLANT for selectability. Adding a third name must not turn the
    validation into a pass-through: `PretrainedConfig` stores an unknown
    keyword and never reads it, which is how `"operator": "sgaet"` would train
    the other operator for a week with nothing raised."""
    with pytest.raises(ValueError, match="unknown operator"):
        _cfg(operator="smprme")


def test_the_gate_heads_reach_the_operator():
    """PLANT for selectability, and the one this arm has failed before.

    `V16_ARM_SMPRIME.md` section 0 records `label_cell` computing a `route` and
    never passing it to `readout`, so the mutilation was silently discarded. A
    head that is computed and dropped on the floor is the same defect: the LM
    would build, forward, train, and be a gate-free operator wearing the arm's
    name. Perturbing the magnitude head alone must move the logits.
    """
    model = _lm(operator="smprime").eval()
    x = _tokens()
    with torch.no_grad():
        before = model(input_ids=x).logits.clone()
        model.model.layers[0].self_attn.m_head.bias.add_(0.5)
        after = model(input_ids=x).logits
    assert not torch.equal(before, after), (
        "the magnitude head does not reach the operator: the arm is wired as a "
        "gate-free operator wearing the arm's name")


def test_the_arm_refuses_a_padding_mask_instead_of_ignoring_it():
    """`ceq/arm_smprime.py::operator` takes no attention mask, and adding one
    would be a new construction in a module this node may not edit. So the
    seam RAISES. The cost this buys back is the one this repository has paid
    twice -- a constraint reported applied and not applied."""
    model = _lm(operator="smprime")
    x = _tokens()
    with pytest.raises(NotImplementedError, match="padding mask"):
        model(input_ids=x, attention_mask=torch.ones_like(x))


def test_the_arm_survives_save_pretrained_and_from_pretrained(tmp_path):
    """Q3's deliverable is a CHECKPOINT, so the operator has to be IN it.

    The plant is the one that matters: rewrite the saved `config.json` to name
    the other operator and require the reloaded model to compute something
    else. If it did not, the operator would be baked into the weights rather
    than read from the config, and a checkpoint could be loaded under the wrong
    operator with every shape matching and nothing raised -- which is exactly
    the hazard `CEQConfig`'s first guard was written for.
    """
    import json

    model = _lm(operator="smprime").eval()
    x = _tokens()
    with torch.no_grad():
        before = model(input_ids=x).logits.clone()
    model.save_pretrained(tmp_path)

    from ceq.hf.modeling_ceq import CEQForCausalLM
    back = CEQForCausalLM.from_pretrained(tmp_path).eval()
    assert back.config.operator == "smprime"
    assert (back.config.smp_beta, back.config.smp_qk, back.config.smp_g) == (1.0, 1.0, 1.0)
    assert (back.config.rho, back.config.lam, back.config.hops) == (None, None, None)
    with torch.no_grad():
        assert torch.equal(before, back(input_ids=x).logits)

    cfg = json.loads((tmp_path / "config.json").read_text())
    cfg.update(operator="sgate", rho=1.5, lam=0.10, hops=2,
               smp_beta=None, smp_qk=None, smp_g=None)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    wrong = CEQForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        assert not torch.equal(before, wrong(input_ids=x).logits), (
            "the operator is not read from the config on load")


# --------------------------------------------- 2. THE DEFAULT PATH IS UNMOVED

def _default_logits(module, seed=0):
    torch.manual_seed(seed)
    model = module.CEQForCausalLM(module.CEQConfig(**TINY)).eval()
    with torch.no_grad():
        return model, model(input_ids=_tokens()).logits


def test_the_default_path_is_bitwise_the_pre_change_code(base_pkg):
    """Same seed, same construction order, `torch.equal` on both the weights
    and the logits. Not `allclose`: a plumbing change that moves the default
    path by 1e-16 is not plumbing."""
    from ceq.hf import modeling_ceq

    old_model, old = _default_logits(base_pkg)
    new_model, new = _default_logits(modeling_ceq)

    o, n = old_model.state_dict(), new_model.state_dict()
    assert sorted(o) == sorted(n), set(o) ^ set(n)
    for k in o:
        assert torch.equal(o[k], n[k]), k
    assert torch.equal(old, new), (
        "the default forward moved: max |delta| = {:.6e}".format(
            float((old - new).abs().max())))


def _one_ulp_up(t):
    return torch.nextafter(t, torch.tensor(float("inf"), dtype=t.dtype))


def test_one_ulp_in_the_operator_is_caught_by_the_bitwise_comparison(base_pkg):
    """PLANT for the bitwise proof, placed INSIDE the operator.

    `lam` is read by `sgate_operator` and by nothing else, so moving it by one
    float32 ulp perturbs exactly the thing the comparison above is guarding.
    Measured on this box: `max |delta| = 5.960e-08`, caught.

    WHAT THIS PLANT ALSO MEASURED, AND IT IS A LIMIT ON THE PROOF ABOVE rather
    than on the change. `torch.equal` on fp32 logits at this shape does NOT see
    one ulp on `model.layers.0.self_attn.qkv.weight[0, 0]` -- nor 64 ulps: the
    operator renormalizes and the perturbation rounds away before the logits.
    It sees one ulp on `lam`, on `lm_head.weight`, on `embed_tokens.weight` and
    on `o_proj.weight`, and 16 ulps on `layers.1.self_attn.qkv.weight`. So the
    bitwise test proves the default path is bit-identical; it does not prove
    that EVERY sub-ulp perturbation anywhere would have been visible.
    """
    from ceq.hf import modeling_ceq

    _, old = _default_logits(base_pkg)

    torch.manual_seed(0)
    lam = float(_one_ulp_up(torch.tensor(0.10)))
    moved_op = modeling_ceq.CEQForCausalLM(
        modeling_ceq.CEQConfig(operator="sgate", lam=lam, **TINY)).eval()
    with torch.no_grad():
        out_op = moved_op(input_ids=_tokens()).logits
    assert not torch.equal(old, out_op), (
        "one ulp on `lam` did not move the logits: the bitwise test above "
        "cannot see the operator and is vacuous")

    new_model, _ = _default_logits(modeling_ceq)
    with torch.no_grad():
        w = new_model.lm_head.weight
        w[0, 0] = _one_ulp_up(w[0, 0])
        out_w = new_model(input_ids=_tokens()).logits
    assert not torch.equal(old, out_w), (
        "one ulp on one weight did not move the logits")


# ------------------------------------------------------------ 3. MATCHED PARAMS

#: Per attention block, the arm carries the parametrization
#: `ceq/arm_smprime.py::ArmSMPrime` carries and a softmax block does not: two
#: per-position scalar heads `nn.Linear(d, 1)` and the three scalar switches
#: `beta`, `qk`, `g`. `V16_ARM_SMPRIME.md` section 1 records the same excess as
#: `4,806 - 4,769 = 37` at `d_model = 16`, which is `2*16 + 5`.
def _expected_excess(d, layers):
    return layers * (2 * (d + 1) + 3)


def _param_accounting(d=TINY["hidden_size"], d_ctrl=None):
    arm = _lm(operator="smprime", hidden_size=d)
    ctrl = _lm(operator="sgate", hidden_size=d if d_ctrl is None else d_ctrl)
    return _n_params(arm), _n_params(ctrl)


def test_the_arm_and_the_softmax_shaped_control_differ_by_named_parameters(capsys):
    """The control is the SAME LM at the same shape on the existing operator:
    `CEQAttention` carries the qkv and output projections a softmax block
    carries and the operator itself has no parameters, which is what makes the
    comparison possible at all.

    THE COUNTS ARE NOT EQUAL AND THE TEST DOES NOT PRETEND THEY ARE. The excess
    is the arm's own parametrization, named parameter by parameter; the
    assertion is that NOTHING ELSE is in it.
    """
    arm, ctrl = _param_accounting()
    excess = _expected_excess(TINY["hidden_size"], TINY["num_hidden_layers"])
    print("\n  arm  (operator=smprime) parameters = {:,}".format(arm))
    print("  ctrl (operator=sgate)   parameters = {:,}".format(ctrl))
    print("  difference = {:,}  = n_layers * (2*(d+1) + 3) = {:,}".format(
        arm - ctrl, excess))
    assert arm - ctrl == excess, (arm, ctrl, arm - ctrl, excess)

    #: and the excess is exactly those tensors, by name
    names = {n for n, _ in _lm(operator="smprime").named_parameters()}
    ctrl_names = {n for n, _ in _lm(operator="sgate").named_parameters()}
    extra = names - ctrl_names
    assert extra == {"model.layers.{}.self_attn.{}".format(i, t)
                     for i in range(TINY["num_hidden_layers"])
                     for t in ("m_head.weight", "m_head.bias",
                               "theta_head.weight", "theta_head.bias",
                               "beta", "qk", "g")}, sorted(extra)
    assert not ctrl_names - names, sorted(ctrl_names - names)


def test_a_mismatched_hidden_size_is_caught_by_the_param_accounting():
    """PLANT for the parameter check: build the control one head wider and
    require the accounting to fire. A count comparison that survives a shape
    mismatch is not a count comparison."""
    arm, ctrl = _param_accounting(d=TINY["hidden_size"],
                                  d_ctrl=TINY["hidden_size"] * 2)
    assert arm - ctrl != _expected_excess(TINY["hidden_size"],
                                          TINY["num_hidden_layers"]), (arm, ctrl)


# -------------------------------------------------------- 4. FINITE GRADIENTS

def _forty_steps(*, steps=40, plant_nan_at=None, **cfg):
    """First step at which any gradient goes non-finite, or None.

    `ceq/arm_smprime.py::gradient_finiteness`'s probe on the LM instead of on
    the standalone arm: same budget, same question, and it keeps nothing.
    """
    model = _lm(**cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = _tokens(seed=7)
    for t in range(steps):
        if t == plant_nan_at:
            with torch.no_grad():
                model.model.layers[0].self_attn.qkv.weight[0, 0] = float("nan")
        opt.zero_grad()
        model(input_ids=x, labels=x).loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in model.parameters()):
            return t
        opt.step()
    return None


def test_forty_steps_on_the_arm_have_no_non_finite_gradient(capsys):
    """`V16_ARM_SMPRIME.md` (g): 40 steps complete from the initialiser on 8 of
    8 configurations of the standalone arm. This is the same question asked of
    the wired LM, CPU, tiny shape. It reports FINITENESS and nothing else."""
    first = _forty_steps(operator="smprime")
    print("\n  40-step gradient probe  cpu  float32  operator=smprime "
          " -> first non-finite: {}".format(first))
    assert first is None, "non-finite gradient at step {}".format(first)


def test_a_planted_nan_is_caught_by_the_gradient_check():
    """PLANT for the gradient check. A finiteness probe that cannot see a NaN
    reports `None` forever and proves nothing."""
    first = _forty_steps(operator="smprime", plant_nan_at=3)
    assert first == 3, first
