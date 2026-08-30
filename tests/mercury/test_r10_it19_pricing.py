"""The Phase 1c pricing carries a self-check that nothing would collect.

`scale/r10_it19_pricing.py` ends in an assert-based `demo()` behind a `__main__`
guard -- the exact shape the Health Inspector struck at the Phase 1a boundary
("1 of 11 phase-1a findings has any runnable binding"), and the shape
`tests/loop/test_phase1a_modules_are_bound.py` exists to repair for three earlier
modules. This file gives the it.19 pricing the same treatment at the same time it
is written, rather than after an audit finds it.

Two nodes, and the second is the one that matters. A demo that exits 0 while
checking nothing satisfies the first node perfectly, so each plant deletes ONE
named guarantee in a COPY of the tree and requires the copy to exit non-zero.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
REL = "scale/r10_it19_pricing.py"

#: (id, line as shipped, line with the guarantee removed). Each names a specific
#: claim the report makes, so a demo that has stopped checking that claim is
#: detected rather than assumed.
PLANTS = [
    ("rss-floor-flag",
     "RSS_FLOOR_ABOVE = 49152",
     "RSS_FLOOR_ABOVE = 10 ** 12"),
    ("twin-multiplier-is-a-band",
     "TWIN_MULT_HI = 0.576326 / 0.159293",
     "TWIN_MULT_HI = 2.136828 / 0.916191"),
    ("budget-actually-binds",
     "DECLARED_BUDGET_S = 21600.0",
     "DECLARED_BUDGET_S = 1.0e12"),
    ("s-squared-law-is-measured",
     "C_ROW_STEP_256 = 0.034045 / 64",
     "C_ROW_STEP_256 = 0.0034045 / 64"),
]


def run(path: pathlib.Path, cwd: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(path)], cwd=str(cwd),
                          capture_output=True, text=True, timeout=300)


def test_the_pricing_self_check_passes_as_shipped():
    r = run(ROOT / REL, ROOT)
    assert r.returncode == 0, (
        f"{REL} self-check fails as shipped:\n{(r.stderr or r.stdout)[-1500:]}")
    assert "ALL CHECKS PASSED" in r.stdout, (
        f"{REL} exited 0 without reaching its own OK line; it may have returned "
        f"early rather than checked anything:\n{r.stdout[-800:]}")


@pytest.mark.parametrize("_id,shipped,broken", PLANTS, ids=[p[0] for p in PLANTS])
def test_the_demo_asserts_something(_id, shipped, broken, tmp_path):
    work = tmp_path / "tree"
    shutil.copytree(ROOT / "scale", work / "scale")
    target = work / REL
    src = target.read_text(encoding="utf-8")
    assert shipped in src, (
        f"the planted line {shipped!r} is not in {REL}; this must-fire is aimed "
        "at a line that no longer exists and would pass without testing anything")
    target.write_text(src.replace(shipped, broken), encoding="utf-8")

    r = run(target, work)
    assert r.returncode != 0, (
        f"{REL} still exits 0 with {shipped!r} replaced by {broken!r}, so its "
        f"demo does not actually check that guarantee:\n{r.stdout[-800:]}")
