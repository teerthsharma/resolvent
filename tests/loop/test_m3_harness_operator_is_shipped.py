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
SHIPPED = ("softmax", "sgate", "sgate_w8")


def _inputs():
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(N, S, D, generator=g)
    return x


def _shipped_tensors(q, k, arm):
    """The shipped operator family.

    `sgate` and `sgate_w8` are the SAME function with one argument different.
    That is deliberate: the windowed arm must not be allowed to become a
    separate operator, because then F4's flatness and F4's capability would be
    measured on two different objects -- which is instrument #17 with the parts
    swapped.
    """
    out = {"softmax": bench._softmax_operator(q, k)}
    try:
        out["sgate"] = bench._causal_sgate_operator(
            q, k, rho=M3.SGATE_RHO, lam=M3.SGATE_LAM)
        out[f"sgate_w{M3.W_WINDOW}"] = bench._causal_sgate_operator(
            q, k, rho=M3.SGATE_RHO, lam=M3.SGATE_LAM, window=M3.W_WINDOW)
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


def test_the_windowed_arm_is_sgate_with_a_band_and_nothing_else():
    """F4's arm has landed. This is the assertion the placeholder promised.

    Three things must hold, all by VALUE:
      1. the operator is bitwise the SHIPPED sgate with `window=W_WINDOW` --
         not a new operator, not a re-derivation;
      2. it is genuinely banded, so the arm is not silently the dense arm;
      3. it is NOT bitwise the unbounded sgate, so the band is load-bearing
         rather than decorative at this sequence length.
    """
    assert "windowed_signed" in M3.ARMS
    torch.manual_seed(SEED)
    arm = M3.Arm("windowed_signed", s=S)
    x = _inputs()
    q, k = arm.wq(x), arm.wk(x)
    with torch.no_grad():
        got = arm._operator(q, k)
        shipped = _shipped_tensors(q, k, arm)

    key = f"sgate_w{M3.W_WINDOW}"
    assert torch.equal(got, shipped[key]), (
        f"windowed_signed is not bitwise the shipped sgate at window="
        f"{M3.W_WINDOW}. The windowed arm must be the same operator with one "
        f"argument different, or F4's flatness and F4's capability are being "
        f"measured on two different objects."
    )
    assert not torch.equal(got, shipped["sgate"]), (
        "windowed_signed is bitwise identical to the UNBOUNDED sgate, so the "
        "band does nothing at this length and the arm is the dense arm wearing "
        "F4's name."
    )

    # the band, checked on entries rather than on the mask that built them
    i = torch.arange(S).view(-1, 1)
    j = torch.arange(S).view(1, -1)
    outside = (i - j > M3.W_WINDOW) | (j >= i)
    assert float(got[:, outside].abs().max()) == 0.0, (
        "windowed_signed has nonzero mass outside the causal band; it is not "
        "the bounded receptive field F4 measured."
    )


def test_windowed_hop2_reaches_exactly_two_windows():
    """The bounded receptive field IS the arm's content, so pin its extent.

    `a` is banded to `w`, so `a @ a` must reach `2w` and no further. If hop 2
    ever reaches further, the arm has stopped being windowed and every
    comparison against F4's flat statistic is void. Checked as a VALUE on the
    product, not as a property of the code that built it.
    """
    torch.manual_seed(SEED)
    arm = M3.Arm("windowed_signed", s=S)
    x = _inputs()
    q, k = arm.wq(x), arm.wk(x)
    with torch.no_grad():
        a = arm._operator(q, k)
        hop2 = a @ a

    w = M3.W_WINDOW
    i = torch.arange(S).view(-1, 1)
    j = torch.arange(S).view(1, -1)
    beyond = (i - j > 2 * w) | (j >= i)
    assert float(hop2[:, beyond].abs().max()) == 0.0, (
        f"hop 2 reaches beyond {2 * w} positions; the arm is no longer windowed."
    )
    # and it must actually USE the second window, or hop 2 is doing nothing
    ring = (i - j > w) & (i - j <= 2 * w)
    assert float(hop2[:, ring].abs().max()) > 0.0, (
        f"hop 2 has no mass between {w} and {2 * w} positions, so the second hop "
        f"buys no reach at all and the arm is one hop in disguise."
    )
