"""G0.17 K-COMPAT -- the two-stack bridge, tested where it can be tested.

WHAT THIS FILE CAN AND CANNOT BIND. Every number in round v17-K's local campaign
was measured on `python 3.11.9 / torch 2.5.1+cu121 / transformers 5.3.0`. Kaggle
run 1 reported `python 3.12.13 / torch 2.10.0+cu128`
(`results/kaggle_v17k_output/ceq-v17-k.log`). There is no 3.12/2.10 interpreter
on this box, so **no test here can bind the target stack.** What every test below
binds instead is that `ceq/compat.py` REPORTS WHAT IT OBSERVED rather than what
it was told to expect: each check that could pass vacuously carries a planted
negative that makes the checker fire.

Two tests are stack-conditional by construction and say so in their skip message
rather than passing quietly on a stack they were not written for.
"""
import json
import platform
import sys
import types

import pytest
import torch
import transformers

from ceq import compat


# ----------------------------------------------------------- 1. version record

def test_version_record_reads_the_live_interpreter_and_not_a_constant():
    rec = compat.version_record()
    # The three that decide whether two numbers are comparable at all.
    assert rec["python"] == platform.python_version()
    assert rec["torch"] == torch.__version__
    assert rec["transformers"] == transformers.__version__
    for key in ("python", "torch", "transformers", "torch_cuda", "cuda_available",
                "device_name", "device_capability", "arch_list", "device_usable",
                "device_in_arch_list_verbatim", "cublas_workspace_config",
                "platform", "threads", "reference", "matches_reference"):
        assert key in rec, key
    # Embeddable in a run header or a manifest: this must not raise.
    json.dumps(rec)


def test_the_reference_comparison_fires_on_a_stack_that_is_not_the_reference():
    """MUST-FIRE. A `matches_reference` that is True for every input records
    nothing. Fed the reference itself, then the reference with one field moved.
    """
    ref = compat.REFERENCE
    same = {"python": ref["python"], "torch": ref["torch"],
            "transformers": ref["transformers"]}
    assert compat.matches_reference(same) is True
    for moved in ("python", "torch", "transformers"):
        off = dict(same, **{moved: "0.0.0"})
        assert compat.matches_reference(off) is False, moved


def test_device_usability_is_minor_compatibility_and_not_list_membership():
    """MUST-FIRE, and it fires on THIS box's own GPU.

    `sm_89` (the certified RTX 4060) is absent from torch 2.5.1+cu121's arch
    list yet every CUDA op in this repository runs on it, so a
    `capability in arch_list` gate reports a working device as broken. The P100
    that Kaggle run 1 was handed must still come back False, or the fix would
    have dissolved the check it is a fix for."""
    u = compat.device_usable_from
    # the certified local device against its own torch's list [MEASURED]
    ada = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
    assert u("sm_89", ada) is True
    assert ("sm_89" in ada) is False              # the naive gate disagrees
    # Kaggle run 1's P100 against Kaggle's torch 2.10 arch list [RUN, from the log]
    kaggle = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
    assert u("sm_60", kaggle) is False
    assert u("sm_75", kaggle) is True             # the T4 the run is pinned to
    # a cubin for a HIGHER minor does not run on a LOWER one
    assert u("sm_86", ["sm_89"]) is False
    # a different major is never compatible
    assert u("sm_90", ["sm_86"]) is False
    # three-digit archs parse as major/minor, not as a bare number
    assert u("sm_100", ["sm_100"]) is True
    assert u("sm_100", ["sm_90"]) is False
    assert u(None, ada) is False


def test_the_version_record_reports_this_boxs_gpu_as_usable():
    """The non-degenerate half of the check above: on a box where CUDA demonstrably
    works, the record must not say the device is unusable."""
    rec = compat.version_record()
    if not rec["cuda_available"]:
        pytest.skip("no CUDA device on this box")
    assert rec["device_usable"] is True, rec


# ------------------------------------------- 2. the six private HF attributes

def test_all_six_private_attributes_are_reported():
    rep = compat.transformers_report()
    assert set(rep["private_attrs"]) == set(compat.PRIVATE_ATTRS)
    assert len(compat.PRIVATE_ATTRS) == 6
    for name, entry in rep["private_attrs"].items():
        for key in ("declared_on_base", "base_value", "effective_value",
                    "effective_owner", "honored", "unknown_to_base",
                    "rename_candidates"):
            assert key in entry, (name, key)


def test_the_six_private_attributes_bind_on_this_transformers():
    """[MEASURED] on transformers 5.3.0. On the Kaggle stack the same call is
    the report, not the assertion -- which is why the version is checked first
    and the test skips rather than claiming a stack it did not run on."""
    if transformers.__version__ != "5.3.0":
        pytest.skip("this bind is a measurement of transformers 5.3.0; running "
                    "on {} -- read compat.transformers_report() instead"
                    .format(transformers.__version__))
    rep = compat.transformers_report()
    for name, entry in rep["private_attrs"].items():
        assert entry["unknown_to_base"] is False, name
        assert entry["honored"] is True, (name, entry)
    assert rep["all_honored"] is True


def test_an_attribute_the_base_normalises_away_is_reported_as_not_honored():
    """MUST-FIRE, and the mechanism is real: `__init_subclass__` rewriting a
    subclass's private attribute is exactly how a transformers release changes
    one of these six without renaming it."""

    class Base:
        _supports_sdpa = True

        def __init_subclass__(cls, **kw):
            super().__init_subclass__(**kw)
            if "_supports_sdpa" in cls.__dict__:      # normalise the override away
                delattr(cls, "_supports_sdpa")

    class Sub(Base):
        _supports_sdpa = False

    entry = compat.attr_binding(Base, Sub, "_supports_sdpa")
    assert entry["honored"] is False
    assert entry["effective_owner"] == "Base"
    assert entry["effective_value"] is True


def test_a_renamed_attribute_is_reported_absent_with_its_candidates():
    """MUST-FIRE. The name vanishing from the base is the failure this module
    exists for, and it must come back with the candidate that replaced it rather
    than as a bare False."""

    class Base:
        _supports_sdpa_implementation = False        # the rename

    class Sub(Base):
        _supports_sdpa = False

    entry = compat.attr_binding(Base, Sub, "_supports_sdpa")
    assert entry["unknown_to_base"] is True
    assert entry["declared_on_base"] is None
    assert "_supports_sdpa_implementation" in entry["rename_candidates"]
    # the honest control: a name that IS on the base raises no alarm
    ok = compat.attr_binding(Base, Sub, "_supports_sdpa_implementation")
    assert ok["unknown_to_base"] is False


# ------------------------------------------------- 3. the GenerationMixin path

def test_generation_mixin_resolves_and_names_the_path_it_came_from():
    got = compat.ensure_generation_mixin()
    assert got["error"] is None
    assert got["found_in"] in ("transformers.generation",
                               "transformers.generation.utils", "transformers")
    # `ceq/hf/modeling_ceq.py:94` imports from `transformers.generation`.
    assert got["import_path"] == "transformers.generation"
    from transformers.generation import GenerationMixin
    assert got["class_name"] == GenerationMixin.__name__


def test_the_shim_binds_the_alternative_path_when_the_expected_one_is_empty():
    """MUST-FIRE. Without a planted negative this check passes on 5.3.0 for the
    single reason that 5.3.0 needs no shim -- the vacuous control this repo has
    struck fifteen of."""
    a, b = "ceq_compat_fake_a", "ceq_compat_fake_b"
    sys.modules[a] = types.ModuleType(a)                       # carries no symbol
    sys.modules[b] = types.ModuleType(b)
    sys.modules[b].GenerationMixin = type("GenerationMixin", (), {})
    try:
        got = compat.ensure_generation_mixin(paths=(a, b))
        assert got["shim_installed"] is True
        assert got["found_in"] == b and got["import_path"] == a
        assert sys.modules[a].GenerationMixin is sys.modules[b].GenerationMixin
    finally:
        del sys.modules[a], sys.modules[b]


def test_a_generation_mixin_that_is_nowhere_is_reported_not_invented():
    a = "ceq_compat_fake_empty"
    sys.modules[a] = types.ModuleType(a)
    try:
        got = compat.ensure_generation_mixin(paths=(a,))
        assert got["found_in"] is None
        assert got["shim_installed"] is False
        assert "GenerationMixin" in got["error"]
    finally:
        del sys.modules[a]


# ---------------------------------------------------- 4. the determinism probe

def test_the_determinism_probe_reports_all_three_regimes_per_op():
    out = compat.probe_determinism(device="cpu")
    assert set(out["ops"]) == {"cumprod", "cumsum"}
    for op, regs in out["ops"].items():
        assert set(regs) == {"strict_forward", "strict_backward",
                             "warn_only_backward"}, op
        for name, r in regs.items():
            assert isinstance(r["executable"], bool), (op, name)
            assert (r["error"] is None) == r["executable"], (op, name, r)
    for key in ("device", "torch", "regime", "regime_cumsum", "inherited",
                "matches_inherited", "cublas_workspace_config"):
        assert key in out, key
    json.dumps(out)


def test_the_probe_restores_the_global_determinism_flag():
    """The probe flips a PROCESS-GLOBAL. One that left it flipped would silently
    change every test collected after it."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        compat.probe_determinism(device="cpu")
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is True
    finally:
        torch.use_deterministic_algorithms(False)
    assert torch.are_deterministic_algorithms_enabled() is False


def test_the_regime_is_derived_from_the_observation_not_from_the_torch_version():
    """MUST-FIRE, four ways. A classifier that always answers `WARN_ONLY` would
    reproduce this box's reading on every stack, and would report a torch that
    FIXED the kernel as though it had not."""
    assert compat.regime(True, True, True) == "STRICT"
    assert compat.regime(True, False, True) == "WARN_ONLY"
    assert compat.regime(True, False, False) == "BLOCKED"
    assert compat.regime(False, False, False) == "FORWARD_BLOCKED"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_this_box_measures_the_backward_hole_that_rulings_1_and_9_rest_on():
    """[MEASURED] python 3.11.9 / torch 2.5.1+cu121 / RTX 4060.

    Rulings 1 and 9 rest on `cumsum_cuda_kernel` having no deterministic CUDA
    implementation, which puts the hole in BACKWARD and forces `warn_only=True`
    for training. That is an INHERITED claim everywhere except here, where it is
    re-measured. On torch 2.10 this test skips, and the probe's own
    `matches_inherited` flag carries the answer on that stack."""
    if not torch.__version__.startswith("2.5"):
        pytest.skip("the cumsum_cuda_kernel hole is a torch 2.5.x measurement; "
                    "running torch {} -- probe_determinism()['matches_inherited'] "
                    "is the reading on this stack".format(torch.__version__))
    out = compat.probe_determinism(device="cuda")
    cp = out["ops"]["cumprod"]
    assert cp["strict_forward"]["executable"] is True
    assert cp["strict_backward"]["executable"] is False
    assert "cumsum_cuda_kernel" in cp["strict_backward"]["error"]
    assert cp["warn_only_backward"]["executable"] is True
    assert out["regime"] == "WARN_ONLY"
    assert out["matches_inherited"] is True


# ------------------------------------------------------------- 5. the selfcheck

def test_selfcheck_imports_every_module_the_run_needs():
    out = compat.selfcheck(verbose=False)
    assert set(out["modules"]) == set(compat.REQUIRED_MODULES)
    assert {"ceq.kdata", "ceq.hf.train", "ceq.hf.modeling_ceq", "ceq.autopilot",
            "scripts.k_cert"} <= set(compat.REQUIRED_MODULES)
    for name, ok in out["modules"].items():
        assert ok is True, name
    assert out["forward_backward"]["finite_loss"] is True
    assert out["forward_backward"]["params_with_grad"] > 0
    assert out["version"]["torch"] == torch.__version__
    assert out["determinism"]["regime"] in ("STRICT", "WARN_ONLY", "BLOCKED",
                                            "FORWARD_BLOCKED")


def test_selfcheck_halts_legibly_naming_the_module_and_the_exception(monkeypatch):
    """MUST-FIRE. The point of the self-check is that a stack mismatch costs one
    minute and names itself, instead of dying deep in Q3."""
    monkeypatch.setattr(compat, "REQUIRED_MODULES",
                        tuple(compat.REQUIRED_MODULES) + ("ceq.no_such_module",))
    with pytest.raises(compat.CompatError) as exc:
        compat.selfcheck(verbose=False)
    msg = str(exc.value)
    assert "ceq.no_such_module" in msg
    assert "ModuleNotFoundError" in msg


def test_selfcheck_halts_on_a_forward_that_does_not_run(monkeypatch):
    """MUST-FIRE for the OTHER half: a self-check that only imports would sign
    off a stack whose model cannot take a step."""
    def boom(*a, **kw):
        raise RuntimeError("planted: the forward is broken")
    monkeypatch.setattr(compat, "_tiny_forward_backward", boom)
    with pytest.raises(compat.CompatError) as exc:
        compat.selfcheck(verbose=False)
    assert "planted: the forward is broken" in str(exc.value)
    assert "RuntimeError" in str(exc.value)
