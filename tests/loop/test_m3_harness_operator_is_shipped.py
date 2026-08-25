"""The CAPABILITY harness must measure an operator the module ships.

THE GAP THIS CLOSES — instrument #17, found again, in the one place it matters most.

Round 2 caught `tgate` carrying every M2 and S2 headline while shipping nowhere,
and bound it: `tests/loop/test_measured_operator_is_shipped.py` resolves arm ->
operator from `scale/pivot_probe.py::build_arm`.

**It never covered `scale/m3_capability.py`.** That file is the CAPABILITY
harness — the absolute bar, the negation-scope oracle, the numbers that decide
whether any of this is worth anything — and its only signed arm resolves to

    bench._causal_tgate_operator(q, k, self.g, self.tau)      # pivot_signed

`tgate` appears nowhere in the shipped path. So round 2's M3 readings —
softmax 1.855584, pivot_signed 1.342215, pivot_unsigned 1.956147 — are
instrument-#17 readings, and the bind written to prevent exactly that was
pointed at the other file.

The lesson is not "we missed one." It is that a bind covers the call site it
names and nothing else, so a bind must be written against the QUESTION ("does
any harness measure a non-shipped operator?") rather than against one module.

VALUE-BOUND. Per the round-2 instrument law — nine structure-comparing
instruments gave false readings, three value-comparing ones never have — this
does not grep for the string `tgate`. It builds each arm's operator tensor and
compares it BITWISE against the tensors the shipped operators produce on the
same inputs. A rename, a refactor, or an alias cannot fool it.
"""
from __future__ import annotations

import pytest
import torch

from ceq import bench
from scale import m3_capability as M3

S, D, N = 24, M3.D_MODEL, 2
SEED = 0

#: Operators the module actually ships. `sgate` is the signed one
#: (`ceq/attention.py`); `softmax` is the baseline every table must carry.
#: `tgate` is deliberately absent — that absence is the whole test.
SHIPPED = ("softmax", "sgate")


def _inputs():
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(N, S, D, generator=g)
    return x


def _shipped_tensors(q, k, arm):
    out = {"softmax": bench._softmax_operator(q, k)}
    try:
        out["sgate"] = bench._causal_sgate_operator(q, k, rho=1.5, lam=0.10)
    except Exception as e:                       # signature drift is a failure
        pytest.fail(f"cannot build the shipped sgate operator: {e!r}")
    return out


@pytest.mark.parametrize("kind", list(M3.ARMS))
def test_every_m3_arm_operator_is_one_the_module_ships(kind):
    """Each arm's operator tensor must be bitwise one of the shipped operators."""
    torch.manual_seed(SEED)
    arm = M3.Arm(kind, s=S)
    x = _inputs()
    q, k = arm.wq(x), arm.wk(x)
    with torch.no_grad():
        got = arm._operator(q, k)
        shipped = _shipped_tensors(q, k, arm)
        matches = [name for name, t in shipped.items()
                   if t.shape == got.shape and torch.equal(t, got)]
    assert matches, (
        f"M3 arm {kind!r} produces an operator tensor that is NOT bitwise equal "
        f"to any shipped operator {SHIPPED}. The capability harness is measuring "
        f"something the module does not ship — instrument #17, in the file whose "
        f"numbers decide the round."
    )


def test_the_comparison_can_actually_fail():
    """Must-fire control. A green above must mean 'shipped', not 'blind'.

    Builds `tgate` explicitly — the operator that ships nowhere — and requires
    the bitwise comparison to reject it. Without this, an arm whose operator
    silently returned zeros would pass by accident.
    """
    torch.manual_seed(SEED)
    arm = M3.Arm("pivot_signed", s=S)
    x = _inputs()
    q, k = arm.wq(x), arm.wk(x)
    with torch.no_grad():
        g = torch.sigmoid(torch.randn(S, generator=torch.Generator().manual_seed(1)))
        tgate = bench._causal_tgate_operator(q, k, g, torch.tensor(1.0))
        shipped = _shipped_tensors(q, k, arm)
        matches = [n for n, t in shipped.items()
                   if t.shape == tgate.shape and torch.equal(t, tgate)]
    assert not matches, (
        "tgate compared bitwise-equal to a SHIPPED operator. Either the shipped "
        "operator changed or the comparison is blind; both void the test above."
    )


def test_the_windowed_arm_that_phase_0_needs_does_not_exist_yet():
    """F4's windowed arm is the Phase-0 subject and is absent from the harness.

    `bench._causal_mask(window=w)` exists and F4's flatness was measured on
    `sgate` at w=8 [round-1 archive], but `m3_capability.ARMS` has never carried
    a windowed arm — which is exactly why F4 has never been capability-tested.
    This test records the gap and MUST be updated, not deleted, when the arm
    lands.
    """
    windowed = [a for a in M3.ARMS if "window" in a]
    assert not windowed, (
        f"a windowed arm now exists ({windowed}). Update this test to assert its "
        f"operator is sgate-with-window and that its hop-2 term is dense within "
        f"the band, then re-point Phase 0 at it."
    )
