"""Containment says the family CAN be softmax. Does gradient descent LEAVE it?

RED-FIRST. `lean/CEQ/V16Domain.lean:433 three_corners_containment` puts softmax
inside the family at `beta = 1`. The reading that turns that expressivity result
into a practical one is that an arm started at the corner stays there unless
moving off it helps -- that the corner is at least stationary. The first
assertion written here was exactly that:

    def test_the_softmax_corner_is_stationary_for_the_family():
        d = _one_step()
        assert abs(d) < 1e-8

and it failed, verbatim, at commit b4c6620 on WIN-16QAL06O9GB, python 3.11.9,
torch 2.14.0+cpu, exit 1, under
`python -m pytest tests/curvature/test_the_softmax_corner_is_not_a_fixed_point.py -q`:

    E  AssertionError: the softmax corner is NOT stationary in the family:
    E  dL/dbeta = -1.300871e-01 at beta = 1.0

WHAT THE GRADIENT IS, WHICH IS WHY THE ARM LEAVES. With the gate at its default
the read is `O_i = Z_i^(1 - beta) * (softmax_i @ V)`, so
`dO_i/dbeta = -log(Z_i) * O_i` -- a per-row scalar times the row's own output.
Nothing in it re-ranks the row's mixing weights; it is a learned per-row
confidence. Trained, that is exactly what it buys: over 2,000 Adam steps on
6,148 held-out-clean Lichess games (`python -m ceqjepa.headtohead --seed {5501,7,99}
--steps 2000`, commits 6612cdf / 82eb2d4, same machine) beta leaves 1.0 and lands
at 0.66689, 0.65393 and 0.67208; held-out cross-entropy falls in all three seeds
and held-out exact-move accuracy does not rise in any of them.
"""
import torch

import ceq.arm_smprime as arm
from ceqjepa.headtohead import ARMS, Head, S_LEN, _loss

SEED = 5501
RED_GRAD = -1.300871e-01


def _one_step():
    torch.manual_seed(SEED)
    m = Head(ARMS["arm-beta-free"])
    assert float(m.beta.detach()) == 1.0
    g = torch.Generator().manual_seed(SEED + 1)
    batch = torch.randint(0, 64, (128, S_LEN + 1, 2), generator=g)
    _loss(m, batch).backward()
    return float(m.beta.grad)


def test_the_softmax_corner_is_not_stationary_and_this_is_the_gradient():
    """The RED, asserted as the finding. VARIES: nothing -- one backward at the
    corner. PINS: the seed, the batch, the gate (off), qk, and every weight."""
    d = _one_step()
    assert abs(d) > 1e-3, "the corner was stationary after all: %.6e" % d
    assert abs(d - RED_GRAD) < 1e-6, "%.6e drifted from the recorded RED" % d


def test_the_whole_beta_gradient_is_a_per_row_gain_and_never_a_re_ranking():
    """`dO_i/dbeta = -log(Z_i) * O_i` exactly: the direction the optimiser pulls
    beta in is a per-row scale on the row's own output, carrying no component
    that changes which j the row attends to."""
    g = torch.Generator().manual_seed(SEED)
    kw = dict(generator=g, dtype=torch.float64)
    q, k, v = (torch.randn(3, 10, 8, **kw), torch.randn(3, 10, 8, **kw),
               torch.randn(3, 10, 5, **kw))
    beta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    o = arm.readout(q, k, v, beta=beta).real
    (grad,) = torch.autograd.grad(o.sum(), beta)
    _, mod = arm.numerator(q, k)
    predicted = float((-torch.log(mod.sum(-1, keepdim=True))
                       * arm.readout(q, k, v, beta=0.37).real).sum())
    assert abs(float(grad) - predicted) < 1e-9, (float(grad), predicted)
