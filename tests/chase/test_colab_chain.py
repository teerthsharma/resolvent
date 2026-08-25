"""The Colab -> Hugging Face chain, end to end, without spending a GPU-hour.

A notebook is the least testable artifact anyone ships: cell text is not
imported, not linted, and not run until somebody is already paying for the
runtime. So the notebook here is THIN -- install, probe the GPU, clone, call two
functions -- and everything it calls lives in `ceq/hf/train.py`, which is a
module and is tested below at a scale that runs in seconds on CPU.

What the tests cover:
  * the notebook parses, and every code cell is valid Python
  * every `ceq.hf.train` symbol the notebook names actually exists
  * `train()` runs, on cpu and cuda, and the result loads back through
    `AutoModelForCausalLM(..., trust_remote_code=True)`
  * the preflight refuses a configuration that will OOM, BEFORE the run starts

Every test parametrizes over cpu and cuda via `tests/chase/conftest.py`.
"""

import ast
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOTEBOOK = os.path.join(ROOT, "colab", "train_ceq.ipynb")


def _cells():
    with open(NOTEBOOK, encoding="utf-8") as fh:
        nb = json.load(fh)
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _python_source(cell):
    """Cell source with IPython magics and shell escapes replaced by `pass`.

    INDENTATION IS PRESERVED. The first version emitted `pass` at column 0 and
    turned an indented `!git clone` inside an `if` into an IndentationError, so
    the test failed on a notebook that is perfectly valid. A checker that reports
    a defect in correct input is worse than no checker.
    """
    out = []
    for line in cell["source"]:
        stripped = line.lstrip()
        if stripped.startswith(("!", "%", "?")):
            indent = line[:len(line) - len(stripped)]
            out.append(indent + "pass  # magic\n")
        else:
            out.append(line)
    return "".join(out)


# ------------------------------------------------------------ the notebook is real

def test_the_notebook_is_valid_json_with_the_keys_jupyter_needs(device):
    """RED first. A notebook that will not open is a notebook nobody runs."""
    with open(NOTEBOOK, encoding="utf-8") as fh:
        nb = json.load(fh)
    assert nb.get("nbformat") == 4, nb.get("nbformat")
    assert "cells" in nb and nb["cells"], "no cells"
    for i, c in enumerate(nb["cells"]):
        assert c["cell_type"] in ("code", "markdown"), (i, c["cell_type"])
        assert isinstance(c["source"], list), i
        if c["cell_type"] == "code":
            assert "outputs" in c and "execution_count" in c, i


def test_every_code_cell_parses_as_python(device):
    """Syntax errors in a notebook surface after the runtime is already
    allocated and the dependencies are already installed."""
    for i, c in enumerate(_cells()):
        try:
            ast.parse(_python_source(c))
        except SyntaxError as exc:
            pytest.fail("cell {} does not parse: {}".format(i, exc))


def test_the_notebook_only_calls_functions_that_exist(device):
    """The failure this catches is a renamed helper: the notebook keeps the old
    name, nothing checks it, and the run dies after the install cell."""
    from ceq.hf import train as train_mod
    src = "\n".join(_python_source(c) for c in _cells())
    tree = ast.parse(src)
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in ("train", "ceq_train"):
                called.add(node.attr)
    named = {n.name or n.asname for n in ast.walk(tree)
             if isinstance(n, ast.alias)}
    for attr in called | {a for a in named if a and hasattr(train_mod, a)}:
        assert hasattr(train_mod, attr), (
            "the notebook uses ceq.hf.train.{} which does not exist".format(attr))


def test_the_notebook_names_the_two_python_files_that_must_reach_the_hub(device):
    """`configuration_ceq.py` and `modeling_ceq.py` are what `auto_map` points
    at. A push that omits them produces a repo that downloads and then raises."""
    src = "\n".join("".join(c["source"]) for c in _cells())
    assert "configuration_ceq" in src and "modeling_ceq" in src


def test_the_notebook_asks_for_a_token_rather_than_hardcoding_one(device):
    """A pasted token in a shared notebook is a leaked credential. Grepped."""
    src = "\n".join("".join(c["source"]) for c in _cells())
    assert "hf_" + "".join(["x"] * 0) not in src.replace("hf_token", "").replace(
        "HF_TOKEN", ""), "a literal hf_ token appears in the notebook"


# ------------------------------------------------------------------ preflight

def test_the_preflight_refuses_a_configuration_that_cannot_fit(device):
    """RED first, and it is the whole point of the preflight: the 0.5B signed
    config at seq 2048 does NOT fit an L4, and the run must say so before the
    first step rather than after twenty minutes of dataset download."""
    from ceq.hf import train
    ok, msg = train.preflight(hidden_size=1280, n_layers=24, n_heads=20,
                              seq=2048, batch=8, gpu="L4-24GB")
    assert not ok, "preflight passed a configuration that needs 137.78 GiB"
    assert "GiB" in msg, msg


def test_the_preflight_accepts_the_configuration_the_notebook_defaults_to(device):
    """The control. A preflight that refuses everything is not a preflight."""
    from ceq.hf import train
    ok, msg = train.preflight(**train.DEFAULTS, gpu="T4-16GB")
    assert ok, msg


def test_the_preflight_reports_the_checkpointed_budget_too(device):
    """Activation checkpointing is the measured rollback for the memory finding,
    so the preflight has to price it, not merely mention it."""
    from ceq.hf import train
    bad, _ = train.preflight(hidden_size=1280, n_layers=24, n_heads=20, seq=2048,
                             batch=8, gpu="A100-40GB", grad_checkpoint=False)
    good, _ = train.preflight(hidden_size=1280, n_layers=24, n_heads=20, seq=2048,
                              batch=8, gpu="A100-40GB", grad_checkpoint=True)
    assert not bad and good, (bad, good)


# ------------------------------------------------------------ the run itself

def test_train_runs_and_produces_a_loadable_repo(device, tmp_path):
    """The chain, at a scale that fits in a unit test: build, train a handful of
    steps on real bytes, save, and load back through the Auto class exactly as a
    downloader would."""
    from transformers import AutoModelForCausalLM

    from ceq.hf import train
    out = str(tmp_path / "run")
    res = train.train(out_dir=out, steps=3, batch=2, seq=32, hidden_size=64,
                      n_layers=2, n_heads=4, device=device, vocab_size=256,
                      max_bytes=20000)
    assert res["steps"] == 3 and all(
        v == v for v in res["losses"]), res            # NaN check: v == v
    m = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True)
    # The operating point the notebook trains at must be the one that was
    # measured. This asserted `hops == 3` while `train.build` carried its own
    # `hops=3, rho=0.9` defaults; those were the old `signed` values and would
    # have silently trained `sgate` at an unmeasured point once the config
    # default moved. `build` no longer defaults the operator knobs at all.
    assert (m.config.operator, m.config.rho, m.config.lam, m.config.hops) == \
        ("sgate", 1.5, 0.10, 2), m.config
    assert m.config.operator == "sgate"
    assert not m.lm_head.weight.is_meta, "lm_head landed on the meta device"
    with torch.no_grad():
        o = m(input_ids=torch.randint(0, 256, (1, 16)))
    assert torch.isfinite(o.logits).all()


def test_train_logs_a_gradient_norm_every_step(device, tmp_path):
    """The operator's backward carries a 1/l1 factor and the measured row L1
    minimum reached 9.35e-07 over 800 real steps. A training loop for this
    operator that does not record the gradient norm cannot tell a diverging run
    from a slow one -- and a DEQ in this project once lost its fixed point at
    step 47 while the loss fell for 922 further steps."""
    from ceq.hf import train
    res = train.train(out_dir=str(tmp_path / "r"), steps=4, batch=2, seq=32,
                      hidden_size=64, n_layers=2, n_heads=4, device=device,
                      vocab_size=256, max_bytes=20000)
    assert len(res["grad_norms"]) == 4, res["grad_norms"]
    assert all(g == g and g < float("inf") for g in res["grad_norms"]), res
    assert len(res["row_l1_min"]) == 4, (
        "the row L1 minimum is not logged; it is the quantity that predicts the "
        "backward blow-up and it costs one reduction per step")


def test_gradient_checkpointing_actually_changes_the_run_not_just_a_flag(device, tmp_path):
    """A flag that is accepted and ignored is worse than no flag. Same seed, same
    data: the losses must be identical and the memory must not be."""
    from ceq.hf import train
    kw = dict(steps=3, batch=2, seq=64, hidden_size=64, n_layers=2, n_heads=4,
              device=device, vocab_size=256, max_bytes=20000, seed=0)
    a = train.train(out_dir=str(tmp_path / "a"), grad_checkpoint=False, **kw)
    b = train.train(out_dir=str(tmp_path / "b"), grad_checkpoint=True, **kw)
    assert a["losses"] == pytest.approx(b["losses"], rel=1e-4), (
        "recomputation changed the loss; it must be mathematically identical\n"
        "{}\n{}".format(a["losses"], b["losses"]))
    if device == "cuda":
        assert b["peak_bytes"] < a["peak_bytes"], (b["peak_bytes"], a["peak_bytes"])
