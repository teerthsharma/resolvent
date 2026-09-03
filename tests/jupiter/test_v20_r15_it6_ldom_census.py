"""CEQ v20 R15 it.6 -- L-DOM: binds the six `lean/CEQ/V16Domain.lean`
declaration lines the domain census cites (pathProd_polar, pathProd_abs,
pathProd_eq_zero_iff, lean_log_junk_makes_the_scan_form_silently_false,
no_prefix_scan_represents_a_zero_gate, pathProd_eq_Wp), plus the three other
cited facts the census's second column rests on, so a renumber or an edit
cannot silently break the census without a test failing.

lean/CEQ/V16Domain.lean:105  pathProd_polar
lean/CEQ/V16Domain.lean:121  pathProd_abs
lean/CEQ/V16Domain.lean:129  pathProd_eq_zero_iff
lean/CEQ/V16Domain.lean:147  lean_log_junk_makes_the_scan_form_silently_false
lean/CEQ/V16Domain.lean:165  no_prefix_scan_represents_a_zero_gate
lean/CEQ/V16Domain.lean:176  pathProd_eq_Wp

ceq/arm_smprime.py:429       "96.9%" BED-M annihilation rate quoted in the census
ceq/arm_smprime.py:110       the shipped magnitude's CLOSED [0,1] cap
ceq/kdata.py:473,475,480     the three registered-bed keys (BED_SPECS)

Read-only citations only: each test reads the exact source line the census
names and asserts the expected substring is still on it. No git write,
nothing touches Kaggle.
"""
from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _line(rel_path: str, n: int) -> str:
    """1-indexed source line, matching grep/Read line numbers."""
    return (ROOT / rel_path).read_text(encoding="utf-8").splitlines()[n - 1]


LEAN = "lean/CEQ/V16Domain.lean"

DECLARATIONS = [
    (105, "pathProd_polar"),
    (121, "pathProd_abs"),
    (129, "pathProd_eq_zero_iff"),
    (147, "lean_log_junk_makes_the_scan_form_silently_false"),
    (165, "no_prefix_scan_represents_a_zero_gate"),
    (176, "pathProd_eq_Wp"),
]


@pytest.mark.parametrize("lineno,name", DECLARATIONS)
def test_the_six_census_declarations_are_still_at_their_cited_lines(lineno, name):
    assert name in _line(LEAN, lineno), (lineno, name, _line(LEAN, lineno))


def test_bedm_annihilation_percentage_is_still_at_arm_smprime_429():
    line = _line("ceq/arm_smprime.py", 429)
    assert "96.9%" in line, line


def test_shipped_magnitude_cap_is_still_closed_zero_one_at_arm_smprime_110():
    line = _line("ceq/arm_smprime.py", 110)
    assert "clamp(u, 0, 1)" in line and "CLOSED" in line, line


def test_registered_bed_keys_are_still_at_their_cited_kdata_lines():
    assert '"bed_m"' in _line("ceq/kdata.py", 473)
    assert '"bed_k"' in _line("ceq/kdata.py", 475)
    assert '"bed_1"' in _line("ceq/kdata.py", 480)
