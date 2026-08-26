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

__all__ = ["d_H", "delta_hat", "n_parts", "parts", "kappa_cert",
           "one_minus_kappa", "neumann_terms"]


def _in_open_cone(v: torch.Tensor) -> bool:
    """Every entry strictly positive and finite. NaN is not in the cone."""
    return bool(torch.isfinite(v).all() and (v > 0).all())


def d_H(p: torch.Tensor, q: torch.Tensor) -> float:
    """Hilbert projective distance, as a difference of logs.

    FINITE WITHIN A PART, +inf ACROSS PARTS. Two vectors lie in the same part of
    the cone exactly when they have the same SUPPORT, and restricted to that
    support both are strictly positive, so the log-ratio oscillation is finite.
    Only a change of support puts them at infinite distance.

    REPAIRED at round 6 iteration 4. The earlier version demanded the strict
    interior and returned +inf whenever any coordinate was zero — including when
    BOTH vectors shared that zero, which is the same part and a perfectly finite
    distance. The defect surfaced from a cross-check against an independently
    written second implementation, which read 1.503823 where this read inf; the
    disagreement was reported rather than reconciled, and this side was wrong.
    It matters because every masked attention row has zeros: under the old
    reading no two causal rows were ever a finite distance apart.

    A negative coordinate is not a cone point and still gives +inf, as does a
    vector with empty support, which belongs to no part and cannot be normalised.
    """
    p = p.reshape(-1).double()
    q = q.reshape(-1).double()
    if p.numel() != q.numel():
        raise ValueError(f"length mismatch: {p.numel()} vs {q.numel()}")
    if not (bool(torch.isfinite(p).all()) and bool(torch.isfinite(q).all())):
        return math.inf
    if bool((p < 0).any()) or bool((q < 0).any()):
        return math.inf                       # not a cone point at all
    sp, sq = (p > 0), (q > 0)
    if not bool((sp == sq).all()) or not bool(sp.any()):
        return math.inf                       # DIFFERENT parts, or no support
    lr = torch.log(p[sp]) - torch.log(q[sp])   # finite WITHIN the shared part
    return float(lr.max() - lr.min())


def parts(rows: torch.Tensor) -> dict[tuple[bool, ...], list[int]]:
    """Group row indices by SUPPORT. Two vectors lie in the same part of the cone
    exactly when they have the same support, and the Hilbert metric is finite
    only within a part."""
    rows = rows.reshape(rows.shape[0], -1).double()
    out: dict[tuple[bool, ...], list[int]] = {}
    for i in range(rows.shape[0]):
        key = tuple(bool(v) for v in (rows[i] > 0).tolist())
        out.setdefault(key, []).append(i)
    return out


def delta_hat(rows: torch.Tensor) -> float:
    """Projective diameter, as a sup over SAME-PART pairs. Reported in nats.

    REPAIRED at round 6 iteration 4, on a disagreement between this module and
    `scale/foreman_hilbert.py` that read 1.503823 against inf on identical input.
    The earlier version returned +inf as soon as any row left the open cone. That
    is not the theorem's Delta. Lemmens-Nussbaum Thm 2.9 states

        Delta(L) = sup{ d(Lx, Ly) : x, y in C with Lx ~_K Ly }

    and the restriction sits on DELTA, not on d. `d_H` stays strict — it is a
    metric on each part, extended by +inf between parts, and that is unchanged.
    But Delta is a supremum taken ONLY over pairs whose images share a part, so
    it can be finite while other pairs in the cone sit at infinite distance.

    Why this is not a technicality: causal rows at different indices ALWAYS have
    different supports, so the unrestricted reading is +inf on every draw. It was
    measured at +inf in 30/30 live cells while the same-part reading gave
    148.8022 to 403.5583 nats. An unrestricted K-A would fire always, forever,
    carrying no information — the vacuous-control class this project has already
    struck five times.

    A row that is entirely zero has empty support and belongs to no part; it is
    excluded rather than counted, because it cannot be normalised and cannot be a
    pivot. `n_parts` is what a caller checks to see whether the map landed its
    image in one part, which is the question K-A actually asks.
    """
    rows = rows.reshape(rows.shape[0], -1).double()
    if not bool(torch.isfinite(rows).all()) or bool((rows < 0).any()):
        return math.inf                      # a negative entry is not a cone point
    best = 0.0
    for key, idx in parts(rows).items():
        if not any(key) or len(idx) < 2:     # empty support, or nothing to pair
            continue
        sub = rows[idx][:, list(key.index(True) for _ in range(0))] if False else rows[idx]
        mask = torch.tensor(key)
        lr = torch.log(sub[:, mask])
        diff = lr.unsqueeze(0) - lr.unsqueeze(1)
        best = max(best, float((diff.amax(-1) - diff.amin(-1)).max()))
    return best


def n_parts(rows: torch.Tensor) -> int:
    """How many parts of the cone the rows occupy, ignoring all-zero rows.

    K-A asks whether T lands its image in a SINGLE part. More than one part means
    the map does not have a well-defined projective diameter over its image, and
    that is a design fault to report rather than a contraction to certify.
    """
    return sum(1 for key in parts(rows) if any(key))


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
