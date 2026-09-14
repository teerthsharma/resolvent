"""Containment is proved for a forward pass. Does it hold through an OPTIMISER?

`lean/CEQ/V16Domain.lean:433 three_corners_containment` is a statement about the
operator's VALUES. A head-to-head is a statement about a training trajectory,
and the two are not the same claim: an operator equal to softmax at every input
can still be reached by a different gradient and land somewhere else, because
the backward of `exp(w)/Z` and the backward of `softmax(w)` are different
programs with different round-off.

This binds the gap. Two heads, one running torch's `F.softmax(...) @ v` and one
running `ceq/arm_smprime.py:254 readout` at `beta = 1` with the gate off, are
built from one seed, fed one batch sequence, and stepped by one optimiser. If
containment survives training their parameters stay together.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from ceqjepa.headtohead import ARMS, Head, S_LEN, n_params, _loss

SEED = 5501
STEPS = 25
BATCH = 32
FORWARD_TOL = 1e-14
TRAJECTORY_TOL = 1e-9


def _batch(seed=SEED):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 64, (BATCH, S_LEN + 1, 2), generator=g)


def _fresh(name, seed=SEED):
    torch.manual_seed(seed)
    return Head(ARMS[name])


def test_the_two_arms_start_from_bitwise_identical_shared_weights():
    """L-NULL's pin, asserted rather than commented: the only thing that differs
    between these arms is the mixing operator."""
    a, b = _fresh("softmax"), _fresh("arm-beta1")
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), n1
    assert n_params(a) == n_params(b) == 26752


def test_the_shipped_operator_is_the_softmax_head_on_one_forward_pass():
    a, b = _fresh("softmax"), _fresh("arm-beta1")
    x = a.emb_from(_batch()[:, :-1, 0])
    x = torch.cat([x, a.emb_to(_batch()[:, :-1, 1])], -1) + a.pos(torch.arange(S_LEN))
    worst = float((a.mix(x) - b.mix(x)).abs().max())
    assert worst < FORWARD_TOL, "forward drift %.6e" % worst


@pytest.mark.parametrize("arm_b", ["arm-beta1", "arm-gate-dead"])
def test_containment_holds_through_25_adam_steps(arm_b):
    """`arm-beta1` is the corner itself; `arm-gate-dead` is the corner reached
    through the critical-point init `MISTAKES.md` V-29 kills, whose gate has no
    gradient (see test_the_gate_can_leave_the_softmax_corner.py) and which
    therefore cannot move off it either.

    RE-PRICED. This was parametrised over `arm-gate`, and it passed for a reason
    it did not state: at that time `arm-gate` WAS the dead init. With the bed's
    default gate now the module's `trainable_heads` offset, the same assertion
    against `arm-gate` reads a worst parameter gap of `1.496637e-01` against a
    `1e-09` tolerance -- the measurement below. What this row binds is a DEAD
    gate holding still, which is a fact about the init and never was a fact
    about containment."""
    a, b = _fresh("softmax"), _fresh(arm_b)
    oa = torch.optim.Adam(a.parameters(), lr=3e-3)
    ob = torch.optim.Adam(b.parameters(), lr=3e-3)
    batch = _batch()
    for _ in range(STEPS):
        for m, o in ((a, oa), (b, ob)):
            loss = _loss(m, batch)
            assert torch.isfinite(loss), "%s went non-finite" % m.spec["varies"]
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
    worst = max(float((p1 - p2).abs().max())
                for (n, p1), (_, p2) in zip(a.named_parameters(), b.named_parameters()))
    assert worst < TRAJECTORY_TOL, (
        "%s left the softmax trajectory after %d steps: worst parameter gap "
        "%.6e" % (arm_b, STEPS, worst))


def test_the_free_beta_arm_does_leave_that_trajectory():
    """The control that keeps the two tests above from being vacuous: an arm
    whose beta can move DOES move off it, so the tolerance is measuring
    containment and not a dead optimiser."""
    a, b = _fresh("softmax"), _fresh("arm-beta-free")
    oa = torch.optim.Adam(a.parameters(), lr=3e-3)
    ob = torch.optim.Adam(b.parameters(), lr=3e-3)
    batch = _batch()
    for _ in range(STEPS):
        for m, o in ((a, oa), (b, ob)):
            loss = _loss(m, batch)
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
    assert abs(float(b.beta.detach()) - 1.0) > 1e-4, (
        "beta never moved: %.9f" % float(b.beta.detach()))


def test_the_woken_gate_arm_also_leaves_that_trajectory():
    """THE REPLACEMENT ROUTE, and the proof the repair reaches the BED and not
    only the module: `arm-gate` at the shipped `trainable_heads` offset must
    leave the trajectory its dead twin above cannot leave.

    VARIES: the gate init, `arm-gate-dead` against `arm-gate`. PINS: the seed,
    the batch, the optimiser, the step count, the tolerance, and every shared
    module -- so the gap is the offset's and nothing else's."""
    a, b = _fresh("softmax"), _fresh("arm-gate")
    oa = torch.optim.Adam(a.parameters(), lr=3e-3)
    ob = torch.optim.Adam(b.parameters(), lr=3e-3)
    batch = _batch()
    for _ in range(STEPS):
        for m, o in ((a, oa), (b, ob)):
            loss = _loss(m, batch)
            assert torch.isfinite(loss), "%s went non-finite" % m.spec["varies"]
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
    worst = max(float((p1 - p2).abs().max())
                for (_, p1), (_, p2) in zip(a.named_parameters(), b.named_parameters()))
    assert worst > TRAJECTORY_TOL, (
        "arm-gate held the softmax trajectory to %.6e after %d steps: the gate "
        "is still dead in the bed" % (worst, STEPS))
    assert float(b.m_head.weight.abs().max()) > 0.0, "the magnitude head never moved"
