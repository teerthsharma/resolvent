"""The Hub package: can somebody actually load and run this checkpoint.

`test_hf_shipping.py` measured what breaks with the merged Triton kernel behind
custom modeling code. This file is the answer to it -- a package that ships
`configuration_ceq.py` + `modeling_ceq.py` with `auto_map`, loads under
`trust_remote_code=True`, and RUNS ON CPU.

THE CPU FALLBACK IS NOT A COURTESY. Measured and already on record: PyPI ships
`triton` wheels for Linux x86_64/aarch64 only, and Triton needs compute
capability 8.0+, so a free-tier Colab T4 (sm_75) cannot run it. A top-level
`import triton` in the modeling file makes the checkpoint fail to LOAD -- not
merely to run slowly -- on Windows, macOS, and every CPU-only box.

Every test parametrizes over cpu and cuda via `tests/chase/conftest.py`.
"""

import json
import os
import shutil
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

transformers = pytest.importorskip("transformers")

HF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "ceq", "hf")


def _tiny_config():
    from ceq.hf.configuration_ceq import CEQConfig
    # `operator="signed"` is NAMED, not inherited. Since 2026-08-25 the default
    # is `sgate` and setting `hops` without naming an operator is refused -- that
    # guard exists precisely so a pre-change config.json cannot load as the wrong
    # operator. These tests were written about the L1 operator, so they say so.
    return CEQConfig(vocab_size=256, hidden_size=64, num_hidden_layers=2,
                     num_attention_heads=4, max_position_embeddings=32,
                     operator="signed", hops=3)


@pytest.fixture()
def saved_repo(tmp_path):
    """A real `save_pretrained` into a directory, exactly as `push_to_hub` does
    it. Nothing here is hand-written JSON -- if `register_for_auto_class` was not
    called, the .py files do not appear and the tests below fail."""
    from ceq.hf.modeling_ceq import CEQForCausalLM
    m = CEQForCausalLM(_tiny_config())
    d = str(tmp_path / "repo")
    m.save_pretrained(d)
    return d


# ------------------------------------------------- the operator does not drift

def test_the_shipped_operator_is_bitwise_the_one_that_was_tested(device):
    """RED first. `modeling_ceq.py` has to be SELF-CONTAINED -- the Hub copies
    the modeling file and its relative imports, and `from ceq.attention import
    ...` is an absolute import of a package that will not be there. So the
    operator is duplicated, and a duplicated operator drifts.

    This is the guard. Bitwise, not 'close'.
    """
    from ceq.attention import ceq_operator
    from ceq.hf.modeling_ceq import ceq_operator as shipped
    torch.manual_seed(0)
    q = torch.randn(2, 4, 24, 16, device=device)
    k = torch.randn(2, 4, 24, 16, device=device)
    a, b = ceq_operator(q, k), shipped(q, k)
    assert torch.equal(a, b), (
        "max abs difference {:.3e}; the Hub copy of the operator has drifted "
        "from ceq/attention.py".format(float((a - b).abs().max())))


def test_the_shipped_operator_is_signed_and_strictly_causal(device):
    """The tier-3 property, re-checked on the copy that ships. A non-negative
    operator has a non-negative influence Jacobian in any ordered semiring, so
    losing the sign here silently drops the module to tier 2."""
    from ceq.hf.modeling_ceq import ceq_operator
    torch.manual_seed(0)
    q = torch.randn(1, 2, 32, 16, device=device)
    a = ceq_operator(q, q)
    assert float(a.min()) < 0.0, "operator is not signed"
    assert float(a.triu(0).abs().max()) == 0.0, "operator is not strictly causal"


# ------------------------------------------------------- it loads without CUDA

def test_the_modeling_file_does_not_import_triton_at_module_scope(device):
    """A top-level `import triton` makes the checkpoint fail to LOAD wherever
    triton is absent -- Windows, macOS, and every CPU-only Linux box. Grepped
    rather than trusted, because the failure is at import time and a test that
    imports the module first cannot see it."""
    src = open(os.path.join(HF_DIR, "modeling_ceq.py"), encoding="utf-8").read()
    for line in src.splitlines():
        s = line.strip()
        assert not (s.startswith("import triton") or s.startswith("from triton")), s


def test_the_model_forwards_on_cpu_in_a_subprocess_with_cuda_hidden(device):
    """Not `model.to('cpu')` inside a CUDA-capable process -- a genuinely
    CUDA-less interpreter, because that is the machine that downloads it.

    `CUDA_VISIBLE_DEVICES=""` is the closest available approximation and it is
    the same probe `conftest._has_cuda` was written to survive.
    """
    from _chase_env import run_isolated
    rc, out, err = run_isolated("""
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""   # before torch is imported
        import torch
        assert torch.cuda.device_count() == 0, "CUDA was not actually hidden"
        from ceq.hf.configuration_ceq import CEQConfig
        from ceq.hf.modeling_ceq import CEQForCausalLM
        m = CEQForCausalLM(CEQConfig(vocab_size=256, hidden_size=64,
                                     num_hidden_layers=2, num_attention_heads=4,
                                     max_position_embeddings=32,
                                     operator="signed", hops=3))
        o = m(input_ids=torch.randint(0, 256, (1, 16)))
        assert o.logits.shape == (1, 16, 256), o.logits.shape
        assert torch.isfinite(o.logits).all()
        print("OK", tuple(o.logits.shape))
    """)
    assert rc == 0, "CPU-only forward failed:\n{}\n{}".format(out[-2000:], err[-2000:])


def test_generation_works_on_cpu(device):
    """`transformers` 5.3.0 no longer has `PreTrainedModel` inherit
    `GenerationMixin`; `can_generate()` returns True only when the class
    inherits it directly. A model that trains fine and cannot `.generate()` is
    a checkpoint nobody can use."""
    from ceq.hf.modeling_ceq import CEQForCausalLM
    m = CEQForCausalLM(_tiny_config()).eval()
    assert type(m).can_generate(), "can_generate() is False; GenerationMixin missing"
    out = m.generate(torch.randint(0, 256, (1, 8)), max_new_tokens=4, do_sample=False)
    assert out.shape == (1, 12), out.shape


# ------------------------------------------------------- the auto_map contract

def test_save_pretrained_writes_both_python_files(saved_repo, device):
    """`custom_object_save` copies the defining .py file ONLY when
    `register_for_auto_class` has been called. Without that call `save_pretrained`
    writes weights and config and no code, and the repo is unloadable."""
    have = set(os.listdir(saved_repo))
    for f in ("configuration_ceq.py", "modeling_ceq.py", "config.json"):
        assert f in have, "{} missing from {}".format(f, sorted(have))


def test_the_config_carries_an_auto_map_for_config_and_causal_lm(saved_repo, device):
    """`AutoConfig.from_pretrained` reads `config_dict["auto_map"]["AutoConfig"]`
    and `get_class_from_dynamic_module` splits it on '.' into module and class.
    An entry naming a module that is not in the repo fails at download time, not
    here."""
    cfg = json.load(open(os.path.join(saved_repo, "config.json")))
    am = cfg.get("auto_map", {})
    assert "AutoConfig" in am and "AutoModelForCausalLM" in am, am
    for key, ref in am.items():
        mod, _, cls = ref.rpartition(".")
        assert os.path.exists(os.path.join(saved_repo, mod + ".py")), (key, ref)
        assert cls, ref


def test_the_repo_round_trips_through_auto_model_with_trust_remote_code(saved_repo, device):
    """The whole chain in one assertion: save, load through the Auto class with
    only the directory and `trust_remote_code=True`, and get identical logits.

    Loading through `AutoModelForCausalLM` rather than the class directly is the
    point -- that is the path that exercises `auto_map`, and it is the path every
    downloader takes.
    """
    from transformers import AutoModelForCausalLM
    torch.manual_seed(0)
    x = torch.randint(0, 256, (1, 16))
    from ceq.hf.modeling_ceq import CEQForCausalLM
    ref = CEQForCausalLM.from_pretrained(saved_repo).eval()
    got = AutoModelForCausalLM.from_pretrained(saved_repo, trust_remote_code=True).eval()
    with torch.no_grad():
        a, b = ref(input_ids=x).logits, got(input_ids=x).logits
    assert torch.equal(a, b), float((a - b).abs().max())


def test_the_model_type_does_not_collide_with_a_builtin(device):
    """transformers' docs require `model_type` to differ from every existing one.
    A collision routes the Auto classes to the LIBRARY implementation and the
    custom code is never executed -- no error, wrong model."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from ceq.hf.configuration_ceq import CEQConfig
    assert CEQConfig.model_type not in CONFIG_MAPPING, CEQConfig.model_type


def test_loading_without_trust_remote_code_is_refused_and_that_is_the_cost(device):
    """Asserted POSITIVELY rather than left as a red test. This is a real,
    permanent cost of shipping custom modeling code -- every downloader opts in
    to arbitrary code execution and some organisations block it by policy -- and
    a cost that is asserted is a cost that cannot quietly stop being true."""
    from transformers import AutoModelForCausalLM
    d = tempfile.mkdtemp(prefix="ceq_trc_")
    try:
        from ceq.hf.modeling_ceq import CEQForCausalLM
        CEQForCausalLM(_tiny_config()).save_pretrained(d)
        with pytest.raises(ValueError, match="trust_remote_code"):
            AutoModelForCausalLM.from_pretrained(d, trust_remote_code=False)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# -------------------------------------- registering for auto class breaks tying

def _round_trip_ties(cls_config_pair, tmp, tie):
    """Save a fresh model, load it back, and report (tied, lm_head_is_meta)."""
    from ceq.hf.configuration_ceq import CEQConfig
    from ceq.hf.modeling_ceq import CEQForCausalLM
    cfg = CEQConfig(vocab_size=256, hidden_size=64, num_hidden_layers=2,
                    num_attention_heads=4, max_position_embeddings=32,
                    tie_word_embeddings=tie)
    d = str(tmp)
    CEQForCausalLM(cfg).save_pretrained(d)
    m = CEQForCausalLM.from_pretrained(d)
    tied = m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
    return tied, m.lm_head.weight.is_meta


def test_registering_for_auto_class_does_not_break_weight_tying(device, tmp_path):
    """RED first, and it is a LIBRARY defect, not one of ours.

    `PreTrainedModel.is_remote_code()` in transformers 5.3.0 is literally
    `cls._auto_class is not None` (`modeling_utils.py:4666`) -- so calling
    `register_for_auto_class()`, which is the ONLY way `save_pretrained` copies
    the .py files into the repo, flips the model into the remote-code branch of
    `mark_tied_weights_as_initialized`. That branch strips already-initialized
    keys out of `missing_keys`; `tie_weights` then sees the tied target as
    'present in the checkpoint', refuses to tie, and `lm_head.weight` is left on
    the META device.

    Nothing raises. The control is `LlamaForCausalLM` built the same way, which
    has `_auto_class = None` and ties correctly.

    So the two things a custom architecture on the Hub needs -- shipped modeling
    code and tied embeddings -- are mutually exclusive in this version.
    """
    tied, is_meta = _round_trip_ties(None, tmp_path / "tied", tie=True)
    assert tied and not is_meta, (
        "after save/load with tie_word_embeddings=True: tied={}, lm_head.weight "
        "on meta device={}. transformers {}.".format(
            tied, is_meta, transformers.__version__))


def test_the_llama_control_ties_correctly_in_this_transformers(device, tmp_path):
    """Calibration for the test above. If Llama also failed, the finding would be
    'tying is broken', not 'register_for_auto_class breaks tying'."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=32,
                      tie_word_embeddings=True)
    d = str(tmp_path / "llama")
    LlamaForCausalLM(cfg).save_pretrained(d)
    m = LlamaForCausalLM.from_pretrained(d)
    assert not m.lm_head.weight.is_meta
    assert m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
    assert LlamaForCausalLM.is_remote_code() is False
    from ceq.hf.modeling_ceq import CEQForCausalLM
    assert CEQForCausalLM.is_remote_code() is True, (
        "register_for_auto_class was not called, so save_pretrained will not "
        "copy the modeling files and the repo will be unloadable")


def test_the_shipped_config_routes_around_the_tying_defect(device, tmp_path):
    """The rollback, asserted. Untied embeddings cost one extra vocab x d matrix
    -- 40.96M parameters on the 515.57M 0.5B configuration, +7.9% -- and remove
    an entire class of silent breakage."""
    from ceq.hf.configuration_ceq import CEQConfig
    assert CEQConfig().tie_word_embeddings is False, (
        "the shipped default still ties; see "
        "test_registering_for_auto_class_does_not_break_weight_tying")
    _, is_meta = _round_trip_ties(None, tmp_path / "untied", tie=False)
    assert not is_meta, "lm_head.weight is still on the meta device after loading"


# --------------------------------------------- training is not on Triton's path

def test_training_the_shipped_model_needs_no_triton_and_no_ampere(device):
    """The finding that unblocks the whole chain, asserted rather than argued.

    `ceq/mz_kernel.py` is FORWARD-ONLY -- it has no backward pass -- so it cannot
    appear in a training step at all. The training path is therefore pure torch,
    which means the sm_80 floor and the Linux-only wheel do NOT gate training,
    and a free-tier Colab T4 can run it.
    """
    from ceq.hf.modeling_ceq import CEQForCausalLM
    m = CEQForCausalLM(_tiny_config()).to(device)
    x = torch.randint(0, 256, (2, 16), device=device)
    out = m(input_ids=x, labels=x)
    out.loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads), "some parameters got no gradient"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
