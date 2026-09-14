"""kappa-controlled reach-avoid beds: the ground on which Theorem 4 can separate.

WHY THIS BED EXISTS. Theorem 4 (v1-M section 4) says a read that composes one
data-dependent matrix per layer computes a degree-<=L polynomial in Q, so

    q - q_hat_L = sum_{h>L} Q^h R = Q^{L+1} (I-Q)^{-1} R
    ||q - q_hat_L||_inf = max_i P_i(tau > L, absorbed in B) <= (1 - 1/kappa)^L

Depth L suffices to error eps iff the absorption-time tail P(tau > L) <= eps, i.e.
L >= kappa ln(1/eps). So separation between an exact solve and a depth-L stack is
IMPOSSIBLE on a low-kappa bed and grows with kappa. That is a prediction about beds,
not about models, and it explains an already-measured tie rather than excusing it:
on the synthetic bed the operator did not separate from a 2,556-parameter MLP
(+0.0264, sd 0.0358, 4/5 seeds, against a frozen >= +0.020 AND 5/5 threshold), and
that bed's kappa is small enough that a shallow baseline is near-exact by Theorem 4.

THE F2 THAT CAME FIRST, kept because the mechanism matters. The first generator set a
per-step absorption mass a = 1/kappa_target and zeroed the diagonal. It did not
control kappa: targets {5, 20, 50, 100} all measured kappa in [3.48, 5.45]. A causal
(lower-triangular) chain with no self-loop DESCENDS monotonically, so tau is bounded
by the descent through n states and not by the absorption rate. The real softmax
operator has P_ii > 0 -- the diagonal is inside the causal mask -- and that is exactly
what lets a state linger. Restoring the self-loop restores control:

    self_loop  0.00   0.50   0.80   0.90   0.95   0.98
    kappa      1.94   3.82   9.46  18.86  37.67  94.08      (n=64, k=3, seed 7, RUN)

SEPARATION HEADROOM, residual as a percentage of the committor's own range (RUN):

    kappa    L=1     L=2     L=4     L=8    L=16
      1.9   21.8%    9.5%    1.6%    0.0%    0.0%    <- no separation is possible here
      9.5   96.0%   83.3%   62.7%   35.1%   12.2%
     37.7  118.8%  114.9%  107.3%   93.7%   71.3%
     94.1  123.7%  122.1%  118.8%  112.6%  101.1%

kappa is MEASURED per instance and journalled, never assumed from the self-loop knob:
the knob sets it only approximately and the bed reports what it actually built.
"""

import math
import torch

__all__ = ["build", "kappa_of", "tail_at_depth", "committor", "BedSpec"]

DTYPE = torch.float64


class BedSpec:
    """One kappa-controlled instance. Every field is measured, not requested."""

    def __init__(self, Q, R, self_loop, seed):
        self.Q, self.R, self.self_loop, self.seed = Q, R, self_loop, seed
        self.n, self.k = R.shape
        self.kappa = kappa_of(Q)
        self.q = committor(Q, R)
        self.range = float(self.q.max() - self.q.min())

    def depth_for(self, eps):
        """Theorem 4: L >= kappa ln(1/eps) suffices. Returns the sufficient depth."""
        return self.kappa * math.log(1.0 / eps)

    def __repr__(self):
        return ("BedSpec(n=%d k=%d self_loop=%.2f seed=%d | kappa=%.2f range=%.4f)"
                % (self.n, self.k, self.self_loop, self.seed, self.kappa, self.range))


def kappa_of(Q):
    """kappa = ||(I-Q)^{-1}||_inf = max_i E_i[tau]. Raises if I-Q is singular."""
    n = Q.shape[-1]
    eye = torch.eye(n, dtype=Q.dtype, device=Q.device)
    diag = 1.0 - torch.diagonal(Q, dim1=-2, dim2=-1)
    tol = max(1e-12, n * torch.finfo(Q.dtype).eps)
    if bool((diag.abs() < tol).any()):
        raise ValueError("I-Q is singular: a transient state is self-absorbing "
                         "(min |1-Q_ii| = %.3e below %.3e)" % (float(diag.abs().min()), tol))
    return float(torch.linalg.inv(eye - Q).sum(-1).max())


def committor(Q, R):
    n = Q.shape[-1]
    eye = torch.eye(n, dtype=Q.dtype, device=Q.device)
    return torch.linalg.solve(eye - Q, R)


def tail_at_depth(Q, R, L):
    """||q - q_hat_L||_inf, the exact Theorem 4 residual a depth-L stack cannot reach."""
    n = Q.shape[-1]
    eye = torch.eye(n, dtype=Q.dtype, device=Q.device)
    tail = torch.linalg.matrix_power(Q, L + 1) @ torch.linalg.solve(eye - Q, R)
    return float(tail.abs().max())


def build(n=64, k=3, self_loop=0.9, seed=0):
    """Causal substochastic Q with a self-loop, plus absorbing rows R.

    Q is lower triangular INCLUDING the diagonal (the self-loop), which is what the
    causal softmax mask actually admits. Row 0 must absorb: under a causal mask it
    sees only j <= 0, so leaving it transient makes I-Q singular by construction.
    """
    if not 0.0 <= self_loop < 1.0:
        raise ValueError("self_loop must be in [0,1); got %r" % (self_loop,))
    g = torch.Generator().manual_seed(seed)
    idx = torch.arange(n)
    raw = torch.rand(n, n, generator=g, dtype=DTYPE).tril()
    raw[idx, idx] = 0.0
    raw = raw / raw.sum(-1, keepdim=True).clamp_min(1e-12) * (1.0 - self_loop) * 0.5
    raw[idx, idx] = self_loop
    R = torch.rand(n, k, generator=g, dtype=DTYPE)
    rem = (1.0 - raw.sum(-1, keepdim=True)).clamp_min(1e-9)
    R = R / R.sum(-1, keepdim=True) * rem
    raw[0] = 0.0
    R[0] = R[0] / R[0].sum()
    return BedSpec(raw, R, self_loop, seed)


if __name__ == "__main__":
    print("(a) kappa is CONTROLLED by the self-loop knob and MEASURED per instance")
    ks = []
    for sl in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98):
        b = build(self_loop=sl, seed=7)
        ks.append(b.kappa)
        print("    self_loop=%.2f -> kappa=%8.2f   L(eps=0.01)=%7.1f   range=%.4f"
              % (sl, b.kappa, b.depth_for(0.01), b.range))
    assert all(ks[i] < ks[i + 1] for i in range(len(ks) - 1)), "kappa is not monotone in the knob"
    assert ks[-1] / ks[0] > 20, "the knob does not span a useful kappa range"

    print("(b) THEOREM 4 identity: q - q_hat_L == Q^{L+1}(I-Q)^{-1} R, checked against"
          " the explicit partial sum")
    worst = 0.0
    for sl in (0.0, 0.8, 0.95):
        b = build(self_loop=sl, seed=3)
        eye = torch.eye(b.n, dtype=DTYPE)
        for L in (1, 4, 16):
            acc, M = torch.zeros_like(b.R), eye.clone()
            for _ in range(L + 1):
                acc = acc + M @ b.R
                M = M @ b.Q
            resid = float((b.q - acc).abs().max())
            worst = max(worst, abs(resid - tail_at_depth(b.Q, b.R, L)))
    print("    worst |identity - closed form| = %.3e" % worst)
    assert worst < 1e-10, "Theorem 4's identity does not hold numerically"

    print("(c) THE MUST-FIRE. A bed whose kappa is too low CANNOT separate an exact solve")
    print("    from a shallow stack, and this bed refuses to be used for a separation claim.")
    lo = build(self_loop=0.0, seed=7)
    hi = build(self_loop=0.95, seed=7)
    lo_head = 100 * tail_at_depth(lo.Q, lo.R, 8) / lo.range
    hi_head = 100 * tail_at_depth(hi.Q, hi.R, 8) / hi.range
    print("    kappa=%.2f  depth-8 residual = %.2f%% of range  -> SEPARATION IMPOSSIBLE"
          % (lo.kappa, lo_head))
    print("    kappa=%.2f  depth-8 residual = %.2f%% of range  -> separation available"
          % (hi.kappa, hi_head))
    assert lo_head < 1.0, "the low-kappa bed should leave a shallow stack essentially exact"
    assert hi_head > 50.0, "the high-kappa bed should leave a shallow stack far from exact"

    print("(d) REFUSAL: a self-absorbing transient state makes I-Q singular and kappa_of RAISES")
    bad = build(self_loop=0.5, seed=1)
    bad.Q[5, 5] = 1.0
    raised = False
    try:
        kappa_of(bad.Q)
    except ValueError as e:
        raised = True
        print("    raised as required: %s" % str(e)[:96])
    assert raised, "kappa_of returned a number for a singular I-Q"

    print("ALL SELF-CHECKS PASSED")
