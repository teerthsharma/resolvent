"""G0-adjacent: `train()` must build the operator it is asked for.

THE DEFECT THIS PINS. `ceq/hf/train.py::build()` takes `**overrides` and
forwards the operator keys to `CEQConfig` verbatim -- that seam works. But
`train()` called `build(hidden_size=..., n_layers=..., n_heads=..., seq=...,
vocab_size=...)` and accepted no operator argument at all, so EVERY `train()`
run built the config default (`sgate`) no matter what the caller wanted. The
round's Q3 trains `arm_smprime`; with the passthrough missing it would have
trained `sgate` while every log line and the saved `config.json` said so in
small print and nobody read it.

Found by the wiring node (`V17_ARM_WIRING.md`) after it made the operator
selectable from config, which is the change that turned a latent gap into the
one thing blocking Q3.
"""
import json
import os

import pytest
import torch

from ceq.hf import train as T

SHAPE = dict(batch=2, seq=16, hidden_size=32, n_layers=1, n_heads=2,
             vocab_size=256, device="cpu", lr=1e-3, log_every=1000)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    p = tmp_path_factory.mktemp("g12") / "corpus.txt"
    p.write_text("the quick brown fox jumps over the lazy dog. " * 400,
                 encoding="utf-8")
    return str(p)


def _saved_operator(out_dir):
    with open(os.path.join(out_dir, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)["operator"]


def test_train_forwards_the_operator_it_is_given(tmp_path, corpus):
    out = str(tmp_path / "smp")
    torch.manual_seed(0)
    rec = T.train(out_dir=out, steps=1, data_path=corpus,
                  operator="smprime", **SHAPE)
    assert _saved_operator(out) == "smprime", _saved_operator(out)
    assert rec["operator"] == "smprime", rec.get("operator")


def test_the_default_operator_is_unchanged(tmp_path, corpus):
    """Non-degeneracy for the test above: the assertion is only meaningful
    because the default is a DIFFERENT string. If both read `sgate` the first
    test would pass on a build that ignores the argument entirely."""
    out = str(tmp_path / "default")
    torch.manual_seed(0)
    rec = T.train(out_dir=out, steps=1, data_path=corpus, **SHAPE)
    assert _saved_operator(out) == "sgate", _saved_operator(out)
    assert rec["operator"] == "sgate", rec.get("operator")


def test_an_operator_override_on_resume_is_refused(tmp_path, corpus):
    """A resumed run takes its operator from the checkpoint's own config, so an
    override passed here would be SILENTLY IGNORED -- the exact shape of the
    defect above. Refuse it instead."""
    out = str(tmp_path / "base")
    torch.manual_seed(0)
    T.train(out_dir=out, steps=1, data_path=corpus, operator="smprime", **SHAPE)
    with pytest.raises(ValueError, match="resume"):
        T.train(out_dir=str(tmp_path / "next"), steps=1, data_path=corpus,
                resume_from=out, operator="sgate", **SHAPE)
