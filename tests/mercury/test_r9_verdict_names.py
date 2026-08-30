"""A verdict must name the cells it compared, not always `settled` and `twin`.

`verdict_of` was written for the settled/twin pair and hardcodes those two
strings, so every contrast between any other pair of cells printed `SETTLED
WINS` or `TWIN WINS` regardless of which cells were in it. With four cell
families in the tree that is now actively misleading: this session's
`argmaxste` vs `softmax` contrast rendered as `SETTLED WINS` while containing
neither cell.

`capability_table._verdict` already had the fix. The duplication is collapsed:
`verdict_of` takes optional names and `_verdict` delegates to it.

THE DEFAULTS MUST REPRODUCE THE OLD STRINGS EXACTLY. Published readings quote
`SETTLED WINS` and `TWIN WINS`, so a changed default would silently rewrite
every rendered card and report. (The separately-raised hazard -- that a changed
verdict would invalidate journalled units through `run_bucket`'s bitwise resume
audit -- does not apply: `contrast` is never called inside `_unit` in either
module, and no `results/*.jsonl` contains a `verdict` key at all. The reason to
pin the defaults is the published prose, not the journals.)

    python -m pytest tests/mercury/test_r9_verdict_names.py -q
"""
from __future__ import annotations

import json
import pathlib

import pytest

from scale import capability_table as CT
from scale.m3_synthetic_settled import contrast, verdict_of

ROOT = pathlib.Path(CT.__file__).resolve().parents[1]


def test_the_defaults_reproduce_the_published_strings_exactly():
    assert verdict_of(0.1, 0.2) == "SETTLED WINS"
    assert verdict_of(-0.2, -0.1) == "TWIN WINS"
    assert verdict_of(-0.1, 0.1) == "NO DIFFERENCE"


def test_it_is_still_strict_at_zero():
    """G6: an interval touching zero does not exclude it."""
    assert verdict_of(0.0, 0.2) == "NO DIFFERENCE"
    assert verdict_of(-0.2, 0.0) == "NO DIFFERENCE"


def test_named_cells_replace_the_hardcoded_pair():
    assert verdict_of(0.1, 0.2, "ARGMAXSTE", "SOFTMAX") == "ARGMAXSTE WINS"
    assert verdict_of(-0.2, -0.1, "ARGMAXSTE", "SOFTMAX") == "SOFTMAX WINS"
    assert verdict_of(-0.1, 0.1, "ARGMAXSTE", "SOFTMAX") == "NO DIFFERENCE"


def test_the_capability_table_helper_delegates_rather_than_duplicates():
    """One implementation. Two would drift."""
    for lo, hi in ((0.1, 0.2), (-0.2, -0.1), (-0.1, 0.1), (0.0, 0.2)):
        assert CT._verdict(lo, hi, "A", "B") == verdict_of(lo, hi, "A", "B")


def test_contrast_carries_the_names_through():
    ref = [1.0, 1.0, 1.0, 1.0, 1.0]
    arm = [0.9, 0.9, 0.9, 0.9, 0.9]
    assert contrast(ref, arm, n_boot=500)["verdict"] == "SETTLED WINS"
    named = contrast(ref, arm, n_boot=500, arm_name="TWINROW",
                     ref_name="SOFTMAX")
    assert named["verdict"] == "TWINROW WINS"


def test_no_journal_carries_a_verdict_so_none_can_be_invalidated():
    """Adversarial pass on the hazard this change was warned about."""
    for j in (ROOT / "results").glob("*.jsonl"):
        for line in j.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            assert "verdict" not in rec.get("value", {}), (
                "a journalled unit now carries a verdict: {}".format(j.name))
