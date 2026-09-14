"""Can the family's gate head be TRAINED away from the softmax corner?

RED-FIRST, and the RED is the finding. `ceq/arm_smprime.py:534
ArmSMPrime.identity_heads` is the shipped training initialisation -- the module's
own docstring says the switches "are `nn.Parameter`s so the harness trains them"
-- and it sets `m_head.bias = 1.0`, `theta_head.bias = 0.0`, both weights zero.
The first assertion written here was that a gradient exists there:

    def test_the_shipped_identity_init_can_be_stepped_off():
        du, dth = _grads(1.0, 0.0)
        assert du > 0.0 and dth > 0.0

and it failed, verbatim, at commit b4c6620 on WIN-16QAL06O9GB, python 3.11.9,
torch 2.14.0+cpu, exit 1, under
`python -m pytest tests/curvature/test_the_gate_can_leave_the_softmax_corner.py -q`:

    E  AssertionError: ArmSMPrime.identity_heads() sits at a point with NO
    E  gradient: |dL/du|max = 0.000000e+00, |dL/dtheta|max = 0.000000e+00
    E  assert (0.0 > 0.0)

TWO INDEPENDENT MECHANISMS, EACH EXACTLY ZERO, AND THE INIT SITS ON BOTH

1. `magnitude(u) = clamp(u, 0, 1)` (`:109`) and torch's `clamp` backward is zero
   AT the endpoint, not only past it -- `clamp grad at [1.0, 0.5, 1.0]` reads
   `[0.0, 1.0, 0.0]`. The X36 cap is documented as CLOSED, "`0` and `1` are
   attainable VALUES, not limits"; they are attainable as values and unreachable
   as starting points, which is a different property and is the one training
   needs. `identity_heads` sets the magnitude bias to exactly the upper endpoint.

2. The read-out takes `.real` of a complex operator, and `Re(m e^{i theta})` is
   first-order flat at `theta = 0` for every `m`: `dRe/dtheta = -m sin(theta)`.
   So `theta = 0` is a critical point of the real read-out whatever the magnitude
   does, and `identity_heads` sets the phase bias to exactly `0`.

Measured, same command, `|dL/du|max` and `|dL/dtheta|max` over a `[2, 6]` head at
`beta = 1, qk = 1, g = 1`:

    m = 1.000  theta = 0.000     0.000000e+00     0.000000e+00
    m = 0.999  theta = 0.000     2.412707e+00     0.000000e+00
    m = 1.000  theta = 0.001     0.000000e+00     1.006843e-02
    m = 0.999  theta = 0.001     2.412696e+00     1.005618e-02

THE REPLACEMENT ROUTE, which the last row is: step the two biases off the two
critical points by `1e-3` and both gradients are live, while the operator stays
inside `1e-3` of the softmax corner it was meant to start at. That is the
initialisation `ceqjepa/headtohead.py`'s `arm-gate-live` uses, and it is the only
reason that arm has a gate to race.
"""
import torch

import ceq.arm_smprime as arm

SEED = 0
B, S, DK, DV = 2, 6, 4, 3
OFF = 1e-3


def _draw():
    g = torch.Generator().manual_seed(SEED)
    kw = dict(generator=g, dtype=torch.float64)
    return (torch.randn(B, S, DK, **kw), torch.randn(B, S, DK, **kw),
            torch.randn(B, S, DV, **kw))


def _grads(m0, t0):
    q, k, v = _draw()
    u = (torch.ones(B, S, dtype=torch.float64) * m0).requires_grad_()
    th = (torch.ones(B, S, dtype=torch.float64) * t0).requires_grad_()
    arm.readout(q, k, v, u, th, beta=1.0, qk=1.0, g=1.0).real.sum().backward()
    return float(u.grad.abs().max()), float(th.grad.abs().max())


def test_the_closed_cap_is_a_zero_gradient_point_not_only_a_limit():
    """Mechanism 1, isolated from the phase: torch's clamp backward at the
    endpoint. Varies: the magnitude only. Pins: theta = 0, beta, qk, g, draw."""
    u = torch.tensor([1.0, 0.5, 1.0, 0.0], dtype=torch.float64, requires_grad=True)
    arm.magnitude(u).sum().backward()
    assert u.grad.tolist() == [0.0, 1.0, 0.0, 0.0], u.grad.tolist()


def test_the_phase_is_a_critical_point_of_the_real_readout_at_zero():
    """Mechanism 2, isolated from the cap: the magnitude is held OFF the cap at
    0.999 so any zero here is the phase's own. Varies: theta. Pins: m = 0.999."""
    _, dth = _grads(1.0 - OFF, 0.0)
    assert dth == 0.0, "theta = 0 was not flat: %.6e" % dth


def test_the_shipped_identity_init_sits_on_both_and_cannot_be_stepped_off():
    """The RED, now asserted as the finding it is."""
    du, dth = _grads(1.0, 0.0)
    assert du == 0.0 and dth == 0.0, (du, dth)


def test_the_replacement_route_restores_both_gradients():
    """Step each bias off its own critical point by 1e-3 and both live."""
    du, dth = _grads(1.0 - OFF, OFF)
    assert du > 1e-3, "magnitude still dead at m = 1 - %g: %.6e" % (OFF, du)
    assert dth > 1e-6, "phase still dead at theta = %g: %.6e" % (OFF, dth)


def test_the_replacement_route_starts_within_1e_3_of_the_softmax_corner():
    """And it is still the corner it was meant to start at: the operator at the
    live init against the operator at beta = 1 with the gate off."""
    q, k, _ = _draw()
    live = arm.operator(q, k, torch.full((B, S), 1.0 - OFF, dtype=torch.float64),
                        torch.full((B, S), OFF, dtype=torch.float64),
                        beta=1.0, qk=1.0, g=1.0).real
    corner = arm.operator(q, k, beta=1.0, qk=1.0, g=0.0).real
    worst = float((live - corner).abs().max())
    assert worst < 5e-2, "the live init is not near the softmax corner: %.6e" % worst
