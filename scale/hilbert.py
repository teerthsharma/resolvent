"""The Hilbert projective metric and the Birkhoff contraction constants.

Round 6's certificate rests on this module, and three fellows consume it at
once — Foreman for both kappa estimators, ARM S for settling, Chase for the
kappa back-fit from round 5's residual journal. It is written once so that a
defect is one defect rather than three.

THE METRIC. For p, q in the open positive cone,

    d_H(p, q) = log( max_j (p_j/q_j) / min_j (p_j/q_j) )
              = max_j log(p_j/q_j) - min_j log(p_j/q_j)

The second form is the one implemented. The first overflows for spreads that are
ordinary in a softmax row: two entries separated by e^350 give a literal ratio of
e^700, which is not representable, while the difference of logs reads 700.0
exactly. This is the same failure class as round 5's arccos-near-1 collapse, and
the module is bound to a test that constructs exactly that spread.

WHY PROJECTIVE MATTERS HERE. d_H(p, q) = 0 whenever p and q are parallel, not
only when they are equal — the metric lives on rays through the cone. The
settling map normalises only at the end, so intermediate vectors are off the
simplex, and a scale-sensitive implementation reports a nonzero distance between
two representations of the same point. The invariance is asserted at five decades
of scale rather than assumed.

THE BOUNDARY IS THE KILL. A zero or negative entry leaves the open cone and the
distance is +inf. That is K-A's only trigger, and it is the reason `kappa_cert`
alone must never be used to decide it — see below.

THE FLOAT REPAIR, pre-registered before any datum landed. Birkhoff gives
kappa = tanh(Delta/4). In float64, tanh(Delta/4) reads exactly 1.0 for
Delta >= 76.246190, and the complement 1 - tanh(Delta/4) is already 7.2% high at
Delta = 75 (1.110223e-16 against a 50-digit truth of 1.035111e-16, quantised to
one ULP) and reads exactly 0 from Delta = 100. Since d_H is an oscillation of
log-ratios over softmax rows, Delta ~ 76 nats is a ratio of e^76 = 3.73e+32,
which is ordinary. A kill written as "kappa_cert >= 1" therefore cannot separate
a genuine failure of positivity from a merely large diameter. So:

  * `delta_hat` is the primary quantity and is reported in nats;
  * K-A fires on Delta = +inf alone;
  * `one_minus_kappa` computes 1 - kappa = 2/(e^(Delta/2) + 1) in closed form
    rather than by subtracting from 1, which keeps it representable to
    Delta ~ 1400 where the float route is long dead.

The last point is load-bearing beyond K-A: 1/(1 - kappa) is the Neumann
conditioning in contract 1.2, so a gap that has silently reached zero makes the
truncation bound meaningless rather than merely imprecise.
"""
from __future__ import annotations

import math

import torch

__all__ = ["d_H", "delta_hat", "kappa_cert", "one_minus_kappa", "neumann_terms"]


def _in_open_cone(v: torch.Tensor) -> bool:
    """Every entry strictly positive and finite. NaN is not in the cone."""
    return bool(torch.isfinite(v).all() and (v > 0).all())


def d_H(p: torch.Tensor, q: torch.Tensor) -> float:
    """Hilbert projective distance, as a difference of logs.

    Returns +inf when either vector leaves the open cone, which is the honest
    reading: the metric is defined on rays through the interior, and a boundary
    point is infinitely far from any interior one.
    """
    p = p.reshape(-1).double()
    q = q.reshape(-1).double()
    if p.numel() != q.numel():
        raise ValueError(f"length mismatch: {p.numel()} vs {q.numel()}")
    if not (_in_open_cone(p) and _in_open_cone(q)):
        return math.inf
    lr = torch.log(p) - torch.log(q)
    return float(lr.max() - lr.min())


def delta_hat(rows: torch.Tensor) -> float:
    """Projective diameter over a batch: max over pairs of d_H.

    Reported in NATS and treated as primary. Returns +inf as soon as any row
    leaves the cone, so a single non-positive entry anywhere fires K-A rather
    than being averaged away by the pairs that happen to be well behaved.
    """
    rows = rows.reshape(rows.shape[0], -1).double()
    if not _in_open_cone(rows):
        return math.inf
    lr = torch.log(rows)
    # d_H(p_a, p_b) = max_j(l_aj - l_bj) - min_j(l_aj - l_bj) over all pairs;
    # formed pairwise rather than by a loop, at n^2 memory in the row count.
    diff = lr.unsqueeze(0) - lr.unsqueeze(1)
    return float((diff.amax(-1) - diff.amin(-1)).max())


def kappa_cert(delta: float) -> float:
    """Birkhoff's certified contraction ratio, tanh(Delta/4).

    SATURATES to exactly 1.0 for Delta >= 76.246190 in float64. That saturation
    is expected and is NOT K-A firing; use `delta_hat == inf` for the kill and
    `one_minus_kappa` for anything that needs the gap.
    """
    if math.isinf(delta):
        return 1.0
    return math.tanh(delta / 4.0)


def one_minus_kappa(delta: float) -> float:
    """The gap 1 - kappa, in closed form: 2/(e^(Delta/2) + 1).

    Algebra: 1 - tanh(x) = ((e^x + e^-x) - (e^x - e^-x))/(e^x + e^-x)
                         = 2e^-x/(e^x + e^-x) = 2/(e^(2x) + 1),
    and with x = Delta/4 this is 2/(e^(Delta/2) + 1). Verified against 50-digit
    Decimal to better than 1e-12 relative at Delta = 10 through 1400, where the
    float route 1 - tanh(Delta/4) has read exactly 0 since Delta = 100.
    """
    if math.isinf(delta):
        return 0.0
    h = delta / 2.0
    if h > 700.0:                       # exp overflows just past here
        return 2.0 * math.exp(-h)       # 2/(e^h + 1) ~ 2e^-h to full precision
    return 2.0 / (math.exp(h) + 1.0)


def neumann_terms(delta: float, tol: float = 1e-6) -> int:
    """Smallest N with kappa^N/(1 - kappa) < tol, per contract 1.2.

    Solved in logs so it never forms kappa^N directly. Returns -1 when no finite
    N suffices, which is the state the caller must report rather than silently
    truncating: at Delta = inf the conditioning does not exist.
    """
    gap = one_minus_kappa(delta)
    if gap <= 0.0:
        return -1
    if gap >= 1.0:                      # kappa <= 0; one term already suffices
        return 1
    # log(kappa) = log1p(-gap), NEVER math.log(1.0 - gap). Reconstructing kappa
    # by subtracting from 1 throws away exactly the precision `one_minus_kappa`
    # exists to keep: for Delta = 76.5 the gap is ~8.1e-17, `1.0 - gap` rounds to
    # exactly 1.0, and math.log of that is 0.0 -- a ZeroDivisionError one line
    # later. This function undid the K-A repair one call after it was made, and
    # the pairing against a brute-force loop is what caught it.
    return max(1, math.ceil(math.log(tol * gap) / math.log1p(-gap)))
