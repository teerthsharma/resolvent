"""The capability number, bound to the run that produced it.

Every number this project has is a val-loss ratio or a Jacobian-sign rate.
`arXiv:2605.20798` measured what a val-loss ratio is worth at the scale this
campaign is aimed at: two significant failures land within 2-3% of baseline
validation loss and still drop 6-16 CLIMB-points, and 1.2B improver rank
predicts 3B improver rank at Spearman rho = -0.27. The 3.34% parity target sits
inside that band. So an exact-match capability score is the only kind of
evidence left that means anything here.

WHAT THIS FILE IS FOR. It is not a second copy of the numbers. It reads the
artifact `results/capability.json` written by `ceq/capability.py` and checks
that the verdict follows from the numbers -- so a report cannot drift from its
own data between here and the README. The bars are pre-registered in
`test_cogs_harness.py` (RESOLUTION_FLOOR 0.20, WIN_MARGIN 0.02) and imported
rather than restated, because a bar restated is a bar that can be edited.

Every test parametrizes over cpu and cuda per the standing rule.
"""
from __future__ import annotations

import json
import os

import pytest

from ceq import harness, lm
from test_cogs_harness import RESOLUTION_FLOOR, WIN_MARGIN

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "capability.json")


def _load():
    if not os.path.exists(RESULTS):
        pytest.fail(f"{RESULTS} missing: run `python -m ceq.capability`")
    with open(RESULTS, encoding="utf-8") as fh:
        return json.load(fh)


def _verdict(entry: dict) -> str:
    """The rule, written once. `gated` when the control cannot learn the task
    at this budget, otherwise the sign of the margin against WIN_MARGIN."""
    gate = entry.get("in_distribution")
    if gate is not None and gate["softmax"] < RESOLUTION_FLOOR:
        return "gated"
    margin = entry["arms"]["sgate"]["exact_match"] - entry["arms"]["softmax"]["exact_match"]
    return "sgate_wins" if margin >= WIN_MARGIN else "no_win"


def test_cogs_is_recorded_with_both_arms_and_its_published_reference(device):
    """Three columns or no table: challenger, matched control, published number.

    A two-column result invites the reader to supply the baseline from memory,
    and this project has already had to strike claims that turned out to be
    memories -- the 0.874-against-0.784 Edge Transformer row was one.
    """
    e = _load()["cogs"]
    assert set(e["arms"]) == {"softmax", "sgate"}, sorted(e["arms"])
    assert e["reference"]["source"] == "arXiv:2010.05465", e["reference"]
    assert e["reference"]["score"] == 0.35, e["reference"]
    assert e["arms"]["softmax"]["n_params"] == e["arms"]["sgate"]["n_params"]
    # the recorded parameter count must be the one this box builds today
    built = lm.TinyLM("softmax", seq=harness.COGS_SEQ, vocab=harness.COGS_VOCAB,
                      seed=0).to(device).n_params()
    assert built == e["arms"]["softmax"]["n_params"], (built, e["arms"]["softmax"])


def test_scan_addprim_jump_is_recorded_against_its_nonzero_floor(device):
    """Unlike SCAN-length, this split has a published baseline above zero.

    0.034 +/- 0.020 SEM (arXiv:2107.01366 Table 3, jump column). Two arms near
    zero separate nothing -- `test_arc_reality.py` is the standing lesson -- so
    the split with a nonzero floor is the one where a comparison can land.
    """
    e = _load()["addprim_jump"]
    assert set(e["arms"]) == {"softmax", "sgate"}, sorted(e["arms"])
    assert e["reference"]["score"] == 0.034, e["reference"]
    assert e["reference"]["source"] == "arXiv:2107.01366", e["reference"]
    built = lm.TinyLM("softmax", seq=harness.SEQ,
                      vocab=len(harness.build_vocab("addprim_jump")),
                      seed=0).to(device).n_params()
    assert built == e["arms"]["softmax"]["n_params"], (built, e["arms"]["softmax"])


def test_the_recorded_verdict_follows_from_the_recorded_numbers(device):
    """The verdict is arithmetic over the bars, not a sentence someone wrote.

    A negative result is the honest outcome of this run if the numbers say so,
    and the only way that stays honest is if the label is recomputed here from
    the same pre-registered constants the runner used.
    """
    for split, e in _load().items():
        assert e["verdict"] == _verdict(e), (split, e["verdict"], _verdict(e))


def test_a_result_reports_the_truncation_ceiling_it_was_scored_under(device):
    """Greedy decoding stops at `max_new`, and COGS targets run to 480 tokens.

    Any item whose gold logical form is longer than `max_new` is scored wrong
    no matter what the model emits. That is identical for both arms so it
    cannot favour either, but it caps the achievable score, and a number
    reported without its ceiling reads as a comparison against the published
    0.35 when it is not one.
    """
    for split, e in _load().items():
        assert 0.0 < e["score_ceiling"] <= 1.0, (split, e["score_ceiling"])
        for arm in e["arms"].values():
            assert arm["exact_match"] <= e["score_ceiling"] + 1e-12, (split, arm)


def test_the_cogs_gate_and_the_cogs_result_come_from_one_model(device):
    """The in-distribution split gates the generalization column or nothing does.

    A control that cleared 0.20 on a differently seeded run says nothing about
    the run that produced the headline, so both numbers must carry the same
    seed and the same step count.
    """
    e = _load()["cogs"]
    assert e["in_distribution"]["seed"] == e["seed"], e
    assert e["in_distribution"]["steps"] == e["steps"], e
    assert e["arms"]["softmax"]["eval_split"] == "gen", e["arms"]["softmax"]
