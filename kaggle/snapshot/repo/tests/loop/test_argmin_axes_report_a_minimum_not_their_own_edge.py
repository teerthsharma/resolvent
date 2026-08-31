"""An argmin at the grid edge is the grid's answer, not the model's.

WHAT THIS PROTECTS. `tests/chase/axes.py::axis_rho` asks one question --
"Does the argmin over rho MOVE between two sizes?" -- and
`tests/chase/test_scale_axes.py::test_the_tuned_rho_is_the_same_at_two_scales`
scores it by comparing the two argmins.

At round 10 iteration 2 that test was GREEN and the green meant nothing. The grid
was `(0.9, 1.2, 1.5, 2.0)`, the ratio was monotone decreasing across all four points
at both sizes, and both argmins were therefore pinned to 2.0 -- the last point.
Equality of two argmins at a shared endpoint is guaranteed by monotonicity; it is a
property of where the grid stops, not of where the minimum is. MERCURY recorded it
as instance 11 of the round's mechanism: keyed on a surface proxy (argmin over a
truncated grid) instead of the thing itself (the location of the minimum).

Iteration 3 extended the grid to `(0.9, 1.2, 1.5, 2.0, 3.0, 4.0)` and the curve
turned around hard -- small +12.3% from 2.0 to 4.0, large +29.7%. Both argmins are
now INTERIOR at 2.0 and bracketed on both sides, so the same test is now
informative, and iteration 9's conditioning on this axis was discharged on that
basis.

WHY A GUARD IS STILL WANTED WITH THE TEST GREEN. The grid is a literal at
`axes.py:165` and the expected point count is a second literal at
`test_scale_axes.py::EXPECTED['rho']`. NEPTUNE marked them in-file as "two literals
for one fact". Nothing ties either to the shape of the curve. Shorten the grid --
to save time, to add a size, to resume a partial sweep -- and BOTH literals stay
self-consistent while the argmin slides back to the edge and the equality test goes
green on nothing. The failure is silent and it lands on the one measurement that
unblocked Phase 1.

This test is GREEN today. That is the point: it is a regression guard, not a
finding, and it fails the moment an axis starts reporting its own boundary.

SCOPE. It checks any axis in `scale_axes.jsonl` that has at least three recorded
points at a size -- fewer than three cannot bracket anything, so those are skipped
with the reason stated rather than passed silently. It reads the journal, never
re-runs the sweep, and costs no GPU.
"""
from __future__ import annotations

import collections
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "tests" / "chase" / "scale_axes.jsonl"

#: Below three points an interior minimum is not expressible, so the check would be
#: vacuous rather than passing. Stated, not silently skipped.
MIN_POINTS_TO_BRACKET = 3

#: Axes whose QUESTION is the location of a minimum. Only these can be wrong about
#: an edge; for the others an edge argmin is the correct and expected answer.
#:
#: `axis_rho` asks "Does the argmin over rho MOVE between two sizes?" -- an argmin
#: question, scored by comparing two argmins, and meaningless if either is pinned to
#: the grid's end.
#:
#: `axis_depth` ("d=256 H=4 seq=128 fixed; L = 2,4,8,16"), `axis_seq` and
#: `axis_heads` ask how the ratio SCALES. A monotone trend is the expected shape and
#: puts the argmin at an edge by construction. NEPTUNE measured depth degrading
#: 1.02 -> 1.10 at the parity point, so the depth argmin will sit at L=2 and that is
#: the finding, not a defect.
#:
#: This allowlist was added BEFORE the depth axis ran. Without it, the first depth
#: sweep would have produced a RED from this guard on a correct measurement -- the
#: guard condemning the innocent, which is the direction R10_MECHANISM.md records as
#: the one MISTAKES.md's planted-positive rule cannot catch. The guard was written
#: about that failure and committed it.
ARGMIN_AXES = frozenset({"rho"})


def points_by_axis_and_size() -> dict[tuple[str, str], list[tuple[float, float]]]:
    """{(axis, size): [(x, ratio), ...] sorted by x} from the committed journal."""
    out: dict[tuple[str, str], list[tuple[float, float]]] = collections.defaultdict(list)
    if not JOURNAL.exists():
        return out
    for line in JOURNAL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "axis" in r and "x" in r and "ratio" in r:
            out[(r["axis"], r.get("size", ""))].append((float(r["x"]), float(r["ratio"])))
    return {k: sorted(v) for k, v in out.items()}


def cells() -> list[tuple[str, str]]:
    """Axis/size pairs this guard can rule on: an argmin axis with enough points."""
    return sorted(k for k, v in points_by_axis_and_size().items()
                  if k[0] in ARGMIN_AXES and len(v) >= MIN_POINTS_TO_BRACKET)


def test_the_allowlist_names_axes_that_actually_exist():
    """Premise. An allowlist naming no recorded axis makes every case below vacuous."""
    recorded = {a for a, _s in points_by_axis_and_size()}
    assert recorded, "no axes recorded at all"
    named = ARGMIN_AXES & recorded
    assert named, (
        f"ARGMIN_AXES={sorted(ARGMIN_AXES)} names no axis present in the journal "
        f"{sorted(recorded)}; this guard is inert and should be said to be"
    )


def test_a_trend_axis_is_not_condemned_for_its_edge_argmin():
    """Must-not-fire. A monotone trend axis must be OUT of scope, or the first depth
    sweep produces a RED from this guard on a correct measurement."""
    assert "depth" not in ARGMIN_AXES, "depth is a scaling axis; its argmin is an edge by design"
    assert "seq" not in ARGMIN_AXES and "heads" not in ARGMIN_AXES


def test_the_journal_has_axes_to_check():
    """Must-fire for this file's own premise. Zero cells would pass every case below
    by vacuity -- the absence-not-earned shape MISTAKES.md records as V-7."""
    assert points_by_axis_and_size(), f"{JOURNAL} parsed to no axis records"
    assert cells(), (
        "no axis has 3+ points at any size, so nothing here can bracket a minimum; "
        "this guard is inert and should be said to be, not passed"
    )


def test_the_check_would_catch_a_pinned_edge():
    """Must-not-be-vacuous. A monotone series must be reported as edge-pinned, or the
    assertions below cannot fail and prove nothing."""
    monotone = [(0.9, 1.07), (1.2, 1.04), (1.5, 1.03), (2.0, 1.02)]
    xs = [x for x, _ in monotone]
    argmin_x = min(monotone, key=lambda t: t[1])[0]
    assert argmin_x in (min(xs), max(xs)), (
        "the edge test does not recognise a monotone series as edge-pinned"
    )


@pytest.mark.parametrize("cell", cells())
def test_the_argmin_is_interior_not_at_the_grid_edge(cell: tuple[str, str]):
    """THE GUARD. Green today; red the moment an axis reports its own boundary."""
    axis, size = cell
    pts = points_by_axis_and_size()[cell]
    xs = [x for x, _ in pts]
    best_x, best_ratio = min(pts, key=lambda t: t[1])
    assert best_x not in (min(xs), max(xs)), (
        f"axis {axis!r} at size {size!r}: the argmin is {best_x}, which is the "
        f"{'low' if best_x == min(xs) else 'high'} end of the grid {xs}. "
        f"An argmin at the edge is where the grid stops, not where the minimum is -- "
        f"any test comparing argmins across sizes passes by monotonicity and measures "
        f"nothing. Extend the grid past {best_x} until the ratio rises "
        f"(best so far {best_ratio:.4f}), or state in the reading that the axis is "
        f"unanswered rather than null."
    )
