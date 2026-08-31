"""NaN control slope vs the M2 verdict. THIS FILE IS RED ON PURPOSE.

The first two tests PIN the observed behaviour. The last two assert the SAFE
property -- that a control which was never measured cannot be scored as a
PASSING control -- and they FAIL. They stay failing until `_verdict` grows an
UNEVALUABLE path. A green here would mean the defect was blessed.

An all-zero control series carries no evidence. `loglog_slope` drops every
y <= 0 point, so it returns (nan, 0) -- correct on its own terms, since it
refuses to clamp. `_verdict` then reads that NaN as `k2 = False` and prints
"control decays as the mechanism requires", followed by M2 = GREEN: a control
arm that was never measured is scored as a PASSING control. There is no
UNEVALUABLE/VOID path and nothing raises.

Both functions are imported, never reimplemented.
"""
import math

from scale.m2_units import _verdict
from scale.pivot_probe import loglog_slope

SIZES = [32, 128, 512, 1024, 2048]


def test_all_zero_series_yields_nan_slope():
    """OBSERVED: (nan, 0) -- zero usable points, not a usable fit."""
    slope, npts = loglog_slope(SIZES, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert npts == 0
    assert math.isnan(slope)


def test_nan_control_slope_is_reported_as_a_passing_control(capsys):
    """OBSERVED: a NaN control IS treated as passing, and M2 reads GREEN."""
    _verdict(a=+0.031, b=float("nan"))
    out = capsys.readouterr().out
    assert "slope(c NOT in P) = +nan -> control decays as the mechanism requires" in out
    assert "M2 = GREEN" in out


def test_an_unmeasured_control_must_not_produce_a_green(capsys):
    """SAFE PROPERTY, CURRENTLY RED. `pivot_signed__not_in_P` reads exactly 0.0
    at every s it was run at, so its series carries no fit and no evidence. A
    verdict computed from it must not print GREEN."""
    sl, npts = loglog_slope(SIZES, [0.0] * 5)
    _verdict(+0.031, sl)
    out = capsys.readouterr().out
    assert "M2 = GREEN" not in out, (
        "an all-zero control (slope=%r, npts=%r) produced:%s%s"
        % (sl, npts, chr(10), out))


def test_verdict_rejects_nan_control_explicitly(capsys):
    """SAFE PROPERTY, CURRENTLY RED. A NaN control slope must raise or be
    marked UNEVALUABLE/VOID, never silently scored as a passing control."""
    raised = None
    try:
        _verdict(+0.031, float("nan"))
    except Exception as exc:  # noqa: BLE001
        raised = exc
    out = capsys.readouterr().out
    assert raised is not None or "UNEVALUABLE" in out or "VOID" in out, (
        "NaN control accepted silently. raised=%r%s%s" % (raised, chr(10), out))
