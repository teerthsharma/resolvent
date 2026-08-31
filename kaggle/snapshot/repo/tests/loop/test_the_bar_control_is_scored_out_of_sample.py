"""The bar certifies a task with an in-sample control and gates arms out-of-sample.

THE DEFECT. `scale/negation_scope.py::calibrate_bar` trains the two-feature
positive control on `feats` and then scores it on **the same `feats`**:

    loss = ((net(feats).squeeze(-1) - target) ** 2).mean()      # :1529, trains
    out["trained_two_feature"] = nrmse(net(feats)... , y)       # :1536, scores

Every arm that bar gates is scored on a held-out batch: `run_arm` reports
`eval_nrmse = nrmse(pred_eval, y_eval)` (`m3_capability.py:206`) against a draw
made at `seed + 12345` (`:281`, `:308`).

So the clause `trained_two_feature < 1.0` (`negation_scope.py:1594`), which is
what certifies a task as learnable at all, is satisfied by an IN-SAMPLE reading,
while every arm it admits must clear the same 1.0 OUT-OF-SAMPLE. The control is
solving an easier problem than the one it certifies.

THE REPO ALREADY STATES THE RULE THIS BREAKS, in another gate, in its own words --
`scale/rips_gate.py:163-168`:

    "A held-out split is not decoration. The decoder has up to twenty free
     parameters against a binary label, so an in-sample reading would credit
     memorisation as decoding and the strike below would be unearned."

Two gates ship in one tree and one of them obeys it.

WHAT THIS COSTS, STATED NARROWLY. It cannot make a crossing false: `t*=2` and
`t*=8` cross held-out on eight seeds each, and that is independent of how the bar
was certified. What it can do is certify a task as learnable when no arm can learn
it held-out -- which is the shape of the reading at `t*=32`, where the bar printed
BAR CALIBRATED and softmax read 1.0034 over eight seeds. That block's NOT LEARNABLE
verdict is consistent with EITHER "softmax is the limitation" or "the bar
overstated the task", and nothing in the round currently separates them.

MARS filed this as attack 3 at iteration 4 and scored it "fires as a method
defect; no verdict moves." That was true then. At iteration 17 a NOT LEARNABLE
verdict rests on the bar, so it moves one.

THE ROUTE. One line: score the control on a fresh draw, the way `rips_gate.fit_eval`
already does. Every block would need re-certifying, and `bar_verdict`'s threshold
may need revisiting since a held-out control reads higher than an in-sample one.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scale" / "negation_scope.py"
FUNC = "calibrate_bar"
CONTROL_KEY = "trained_two_feature"


def function_source(path: pathlib.Path, name: str) -> str:
    """The source text of one function, by AST so a rename cannot silently pass."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""
    raise AssertionError(f"{name} not found in {path.name}; this guard reads the "
                         "wrong file or the function was renamed")


def tensors_trained_on(src: str) -> set[str]:
    """Names fed to the net inside the optimisation loop."""
    out = set()
    for line in src.splitlines():
        if "loss" in line and "net(" in line:
            out |= set(_net_args(line))
    return out


def tensors_scored_on(src: str) -> set[str]:
    """Names fed to the net on the line that records the control's reading."""
    out = set()
    for line in src.splitlines():
        if CONTROL_KEY in line and "net(" in line:
            out |= set(_net_args(line))
    return out


def _net_args(line: str) -> list[str]:
    """The identifiers passed to `net(...)` on one line."""
    out, i = [], line.find("net(")
    while i != -1:
        depth, j = 0, i + 3
        for j in range(i + 3, len(line)):
            if line[j] == "(":
                depth += 1
            elif line[j] == ")":
                depth -= 1
                if depth == 0:
                    break
        arg = line[i + 4:j].strip()
        if arg.isidentifier():
            out.append(arg)
        i = line.find("net(", j)
    return out


def test_the_reader_finds_both_sites():
    """Must-fire for this file's own instrument. If either set is empty the
    comparison below passes by vacuity, which is the shape this round catalogues."""
    src = function_source(SOURCE, FUNC)
    trained, scored = tensors_trained_on(src), tensors_scored_on(src)
    assert trained, f"found no `net(...)` call in {FUNC}'s optimisation loop"
    assert scored, f"found no `net(...)` call on {FUNC}'s {CONTROL_KEY} line"


def test_the_reader_can_tell_the_two_apart():
    """Must-fire, other half. A reader that returned the same set for any input
    would report agreement whatever the source said."""
    held_out = (
        "        loss = ((net(feats).squeeze(-1) - target) ** 2).mean()\n"
        '        out["trained_two_feature"] = nrmse(net(feats_eval).squeeze(-1), y_eval)\n'
    )
    assert tensors_trained_on(held_out) == {"feats"}
    assert tensors_scored_on(held_out) == {"feats_eval"}, (
        "the reader cannot distinguish a held-out scoring tensor from the training "
        "one, so its verdict below carries no information"
    )


def test_the_bar_control_is_not_scored_on_the_tensor_it_trained_on():
    """THE DEFECT. RED while the control is in-sample.

    Green once `calibrate_bar` scores `trained_two_feature` on a draw it did not
    optimise against -- which is what `rips_gate.fit_eval` already does for the
    decoder gate in the same tree.
    """
    src = function_source(SOURCE, FUNC)
    trained = tensors_trained_on(src)
    scored = tensors_scored_on(src)
    shared = trained & scored
    assert not shared, (
        f"{FUNC} trains the positive control on {sorted(trained)} and scores it on "
        f"{sorted(scored)} -- sharing {sorted(shared)}. The clause "
        f"`{CONTROL_KEY} < 1.0` certifies a task as LEARNABLE from an IN-SAMPLE "
        "reading, while every arm it gates must clear 1.0 OUT-OF-SAMPLE "
        "(m3_capability.py:206, eval batch at seed+12345). The control is solving "
        "an easier problem than the one it certifies. rips_gate.py:163-168 states "
        "the rule this breaks, in this repo's own words, for the other gate: 'an "
        "in-sample reading would credit memorisation as decoding'. Route: score on "
        "a fresh draw; every block then needs re-certifying and bar_verdict's "
        "threshold may need revisiting, since a held-out control reads higher."
    )


@pytest.mark.parametrize("path,line", [
    ("scale/rips_gate.py", "held-out"),
])
def test_the_sibling_gate_still_does_it_correctly(path: str, line: str):
    """The comparison this finding rests on. If `rips_gate` ever stops holding out,
    the argument above loses its in-repo precedent and this file should say so
    rather than quietly keep citing it."""
    src = (ROOT / path).read_text(encoding="utf-8")
    assert line in src, (
        f"{path} no longer mentions {line!r}; the precedent this guard cites for "
        "the correct behaviour has moved and the citation must be re-checked"
    )
