"""The settling driver: iterate T to its fixed point, journal the Hilbert residual.

ARM S needs something that runs T to convergence and records how it got there.
Foreman owns T's exact FORM; this owns the loop around it, so it is written
generic over T — any callable from the cone to the cone plugs in, and fixing the
form later changes nothing in this file.

WHAT IS JOURNALLED, and why it is the successive distance rather than the distance
to the limit. m* is not known in advance, so the residual recorded at step t is

    r_t = d_H(m_{t+1}, m_t)

Since m_{t+1} = T(m_t) and m_t = T(m_{t-1}), contraction gives r_t <= kappa *
r_{t-1} directly, so the recorded sequence obeys the certified ratio without ever
needing m*. It also gives the step bound contract 1.1 asserts: r_t <= kappa^t * r_0
means the iteration reaches tol by

    t* = ceil( log(r_0 / tol) / log(1/kappa) )

which is the quantity the driver is tested against, as an UPPER bound. Observing
faster contraction than certified is fine; observing slower means the theorem or
the code is wrong, and the test is one-sided for exactly that reason.

FAILURE IS REPORTED, NEVER RETURNED QUIETLY. Two ways a run can fail to settle:
the map does not contract (a permutation is an isometry on the cone and will
happily spin to the step cap), or the iterate reaches the cone boundary and the
residual becomes infinite. Both set `converged = False`, and the second also sets
`left_cone`, because they are different diagnoses and a caller that cannot tell
them apart will misattribute K-A.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import torch

from scale.hilbert import d_H

__all__ = ["SettleResult", "settle", "column_diameter", "predicted_steps"]


@dataclass
class SettleResult:
    m: torch.Tensor                          # the iterate where the run stopped
    residuals: list[float] = field(default_factory=list)
    steps: int = 0
    converged: bool = False
    left_cone: bool = False                  # the iterate reached the boundary

    @property
    def observed_ratios(self) -> list[float]:
        """r_{t+1}/r_t per step. Every entry must sit at or below the certified
        kappa; the maximum is the empirical contraction the run actually showed."""
        return [self.residuals[i + 1] / self.residuals[i]
                for i in range(len(self.residuals) - 1)
                if self.residuals[i] > 0.0]


def settle(T, m0: torch.Tensor, *, tol: float = 1e-12,
           max_steps: int = 1000) -> SettleResult:
    """Iterate `T` from `m0` until the successive Hilbert residual falls below
    `tol`, or until `max_steps`. Never raises on a bad map — it reports."""
    m = m0.reshape(-1).double().clone()
    res = SettleResult(m=m)
    for t in range(1, max_steps + 1):
        nxt = T(m).reshape(-1).double()
        r = d_H(nxt, m)
        res.steps = t
        if not math.isfinite(r):
            # Boundary, or a NaN. Recorded and stopped: an infinite residual
            # appended to the journal would poison every ratio downstream.
            res.left_cone = True
            res.m = nxt
            return res
        res.residuals.append(r)
        m, res.m = nxt, nxt
        if r < tol:
            res.converged = True
            return res
    return res


def column_diameter(a: torch.Tensor) -> float:
    """Projective diameter of a positive matrix acting on the cone.

    For a positive matrix this is the largest Hilbert distance between two of its
    COLUMNS. That characterisation is not cited here; it is checked numerically in
    `tests/loop/test_settle.py` against a direct sample of the supremum over cone
    pairs, which can only be a lower bound and must not exceed it.
    """
    a = a.double()
    n = a.shape[1]
    if n < 2:
        return 0.0
    return max(d_H(a[:, i], a[:, j]) for i, j in itertools.combinations(range(n), 2))


def predicted_steps(r0: float, tol: float, kappa: float) -> int:
    """t* = ceil(log(r_0/tol)/log(1/kappa)), contract 1.1's step bound.

    Returns -1 when kappa >= 1, where the bound does not exist — the caller must
    report that rather than substitute a number. This is the same discipline as
    `hilbert.neumann_terms`, which was where reconstructing a quantity by
    subtraction destroyed it.
    """
    if not (0.0 < kappa < 1.0) or r0 <= tol:
        return -1 if kappa >= 1.0 else 0
    return math.ceil(math.log(r0 / tol) / math.log(1.0 / kappa))
