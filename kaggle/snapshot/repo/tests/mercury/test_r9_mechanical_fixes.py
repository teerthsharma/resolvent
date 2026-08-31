"""Regression battery for the four mechanical defects fixed in R9 iteration 1.

Covers FINDINGS A2 (bucket accounting), A3 (missing difficulty dial), A6 (dead
guard and unreachable tail in the IMPACT builder) and A5 (a computed-and-
discarded value in a shipped gate).

Every test in this file was executed against 74e5590 BEFORE the corresponding
fix landed, and every one failed there; the failure text is quoted in the
iteration report. A test that passes before its fix exists is testing nothing.

Run this file alone:

    python -m pytest tests/mercury/test_r9_mechanical_fixes.py -q
"""
from __future__ import annotations

import inspect
import pathlib

import numpy as np
import pytest
import torch

import scale.etask_k5e as K5E
import scale.impact as IMPACT
import scale.negation_scope as NS
from scale import bucket


# --------------------------------------------------------------------------
# A2 -- run_bucket counts the units it was handed, not the whole journal.
# --------------------------------------------------------------------------
#
# Property: for every unit list U and every journal J, where J may hold records
# belonging to OTHER tasks, the returned accounting satisfies
# 0 <= done <= total == len(U) and remaining == total - done.
#
# Since `--task` landed one journal legitimately holds several tasks, so a count
# taken over the journal is no longer a count of this run's units.

def _units(prefix: str, n: int) -> list:
    return [(f"{prefix}_u{i}", {"i": i}) for i in range(n)]


def _compute(params: dict) -> dict:
    return {"v": params["i"]}


@pytest.fixture()
def journal_dir(tmp_path, monkeypatch):
    """Point the journal at a scratch directory so no shipped result is touched."""
    monkeypatch.setattr(bucket, "RESULTS", tmp_path)
    return tmp_path


def test_accounting_ignores_other_tasks_records(journal_dir, capsys):
    name = "mercury_accounting_full"
    j = bucket.Journal(name)
    for key, params in _units("other_task", 25):
        j.append(key, _compute(params), {})

    mine = _units("my_task", 10)
    acc = bucket.run_bucket(name, mine, _compute, budget_s=60.0, verify=0)

    assert acc["total"] == 10
    assert 0 <= acc["done"] <= acc["total"]
    assert acc["remaining"] == acc["total"] - acc["done"]
    assert acc["done"] == 10
    assert acc["remaining"] == 0

    out = capsys.readouterr().out
    assert "25/10 units already journalled" not in out
    assert "[26/10]" not in out


def test_accounting_still_reports_work_left_when_the_budget_stops_it(journal_dir):
    """The fix must not collapse `remaining` to zero -- a partial run still says so."""
    name = "mercury_accounting_partial"
    j = bucket.Journal(name)
    for key, params in _units("other_task", 25):
        j.append(key, _compute(params), {})

    mine = _units("my_task", 10)
    acc = bucket.run_bucket(name, mine, _compute, budget_s=0.0, verify=0)

    assert acc["total"] == 10
    assert acc["ran_this_bucket"] == 0
    assert acc["done"] == 0
    assert acc["remaining"] == 10


def test_accounting_counts_a_partially_journalled_unit_set(journal_dir):
    """Mixed journal: 4 of this task's own units present, plus 25 foreign ones."""
    name = "mercury_accounting_mixed"
    j = bucket.Journal(name)
    for key, params in _units("other_task", 25):
        j.append(key, _compute(params), {})
    for key, params in _units("my_task", 10)[:4]:
        j.append(key, _compute(params), {})

    mine = _units("my_task", 10)
    acc = bucket.run_bucket(name, mine, _compute, budget_s=0.0, verify=0)

    assert acc["done"] == 4
    assert acc["remaining"] == 6


# --------------------------------------------------------------------------
# A3 -- every registered task has a readable difficulty dial, or an explicit
#       absent one. Neither reader may raise KeyError.
# --------------------------------------------------------------------------
#
# Property: for every task name registered in M3_TASKS, looking up t* returns
# without raising. Tasks that carry no derived dial return None rather than
# exploding, and rather than degrading to a wrong number.

DIAL_LESS = ("impact", "impact_hetero", "e4prime")


def test_every_registered_task_has_a_readable_dial():
    for task in NS.M3_TASKS:
        NS.e_t_star(task, 64)  # must not raise


def test_dial_less_tasks_report_absence_not_a_number():
    for task in DIAL_LESS:
        assert task in NS.M3_TASKS, f"{task} is no longer registered"
        assert task not in NS.E_T_STAR
        assert NS.e_t_star(task, 64) is None


def test_the_dial_is_still_exact_where_it_exists():
    assert NS.e_t_star("e3_t1", 64) == 1
    assert NS.e_t_star("e3_t8", 64) == 8
    assert NS.e_t_star("e3_t32", 64) == 32
    assert NS.e_t_star("e1_anchor", 64) == 63


def test_the_k5e_reader_routes_through_the_guard():
    """Source check: the reader must not index E_T_STAR directly again.

    `etask_k5e.main` trains cells, so it cannot be called from a unit test. This
    is the smallest thing that fails if the unguarded index comes back.
    """
    src = pathlib.Path(K5E.__file__).read_text(encoding="utf-8")
    assert "E_T_STAR[task]" not in src
    assert "e_t_star(task" in src


# --------------------------------------------------------------------------
# A6 -- the IMPACT builder's d-range guard raises like its siblings, and the
#       unreachable tail after `return` is gone.
# --------------------------------------------------------------------------
#
# Property: for every d outside [1, s), make_impact_batch raises ValueError; for
# d inside the range it builds normally.

@pytest.mark.parametrize("d", [0, -1, 1024, 4096])
def test_builder_rejects_out_of_range_d(d):
    with pytest.raises(ValueError):
        IMPACT.make_impact_batch(2, 1024, d)


def test_builder_accepts_an_in_range_d():
    x, y, f, p = IMPACT.make_impact_batch(2, 1024, 24)
    assert x.shape[0] == 2 and y.shape[0] == 2


def test_builder_has_no_unreachable_tail():
    src = inspect.getsource(IMPACT.make_impact_batch)
    assert src.count("return ") == 1, "code after the return is unreachable"
    assert "pass" not in src, "the no-op guard is back"


# --------------------------------------------------------------------------
# A5 -- the decoder gate returns the planted spread it computes, and the
#       non-degeneracy clause actually reads it.
# --------------------------------------------------------------------------
#
# Property: for any draw, `planted_sd` is the smallest per-column spread of the
# planted regressors (excluding the intercept), it is reported, and a draw whose
# planted regressors are constant fails the gate's non-degeneracy clause.

def test_decoder_gate_reports_the_planted_spread():
    r = IMPACT.impact_decoder_gate(n=64, s=64, seed=0)
    assert "planted_sd" in r
    assert r["planted_sd"] > 1e-6
    assert r["pass_nondeg"] is True


def test_planted_spread_is_not_a_copy_of_the_label_spread():
    """The old placeholder was literally `float(y_np.std())` -- label_sd again.

    A non-degeneracy clause that re-reads a quantity already in the clause
    cannot fire independently of it.
    """
    r = IMPACT.impact_decoder_gate(n=64, s=64, seed=0)
    assert r["planted_sd"] != pytest.approx(r["label_sd"])


def test_nondegeneracy_fires_on_constant_planted_regressors(monkeypatch):
    """Adversarial pass: the clause must be able to read FALSE.

    A clause that structurally cannot fail is vacuous before it runs.
    """
    def constant_planted(x, f, p):
        return torch.ones(x.shape[0], 3, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(IMPACT, "impact_planted_features", constant_planted)
    r = IMPACT.impact_decoder_gate(n=64, s=64, seed=0)

    assert r["planted_sd"] <= 1e-6
    assert r["pass_nondeg"] is False
    assert r["passes"] is False


def test_decoder_gate_carries_no_open_questions_in_comments():
    src = inspect.getsource(IMPACT.impact_decoder_gate)
    assert "placeholder" not in src
    assert "?" not in src, "open questions in comment form are still in the gate"
