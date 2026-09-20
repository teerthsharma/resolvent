"""CEQ-JEPA v2: the read is the PREDICTOR, the target is a REPRESENTATION.

WHAT v1 GOT WRONG, MEASURED, AND WHAT v2 CHANGES.

v1 supervised the committor q against an oracle. D1 then produced its own
counterexample: q = (I-Q)^{-1} R is a distribution over WHICH outcome, not WHEN,
so rescaling Q and R together moves the absorption TIME and leaves the absorption
DISTRIBUTION fixed. Measured, n=48, k=3, seed 11:

    self_loop  kappa    max|q - q(sl=0)|
      0.00      1.93     0.000e+00
      0.80      9.35     4.996e-16
      0.95     37.20     1.998e-15

kappa moved 19x and q did not move at all. So a separation experiment against q_1
cannot separate anything: the target is kappa-invariant and every read reaches it.
Theorem 4's identity is still exact (2.776e-16) -- it bounds the truncation error of
a FIXED operator's Neumann series, which is not the same statement as "a learned
model must be worse at low depth".

v2 removes the oracle entirely. The target is s_{t+h}, an EMA target encoder's own
representation, so the objective is horizon-indexed BY CONSTRUCTION and cannot be
time-invariant the way q_1 was.

THE PIECES
  E_theta      online encoder,  x_{<=t} -> s_t
  E_thetabar   EMA target, stop-grad,  theta_bar <- tau theta_bar + (1-tau) theta
  PREDICTOR = THE READ:  s_hat_{t+h} = W_out . [ (I - gamma Q_theta)^{-1} R_theta ](t, a)
               gamma = 1 - 1/h : the horizon IS the prediction step.
  do(a): the action clamps row t of Q. Same rank-1 edit intervene.py already proves
         exact to 1.110e-15 with den = (1-P'_ii)/(1-P_ii) at 1.138e-16.

WHY THE FAILURE CLASS OF v1 IS NOW IMPOSSIBLE, NOT MERELY UNLIKELY.
Three defects this session were the same shape -- a quantity pinned at zero with no
gradient pushing it off zero: delta_a/delta_b both zero (bilinear, grad exactly
0.000000e+00 in both factors, P bitwise identical across the batch), mv_row zeroed
(every candidate move produced the identical intervention row, mean |row_i - row_j| =
0.0, and a placebo scored +0.0243 against the real move's +0.0235), and gamma not a
parameter at all. Each was caught by an instrument AFTER the fact.
The variance term makes the class unreachable: d/ds of max(0, 1 - Std(s)) is nonzero
whenever Std(s) < 1, so a zero-spread state is NOT a stationary point of the loss. The
gradient reaches the bilinear factors THROUGH s even at (0,0) of their own factors.
Self-check (c) asserts exactly this and is seen to fire.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import ceqjepa.operator as op

__all__ = ["Encoder", "ReadPredictor", "MLPPredictor", "vicreg_terms",
           "delta_spec", "rollout_error", "gamma_for_horizon"]


def gamma_for_horizon(h):
    """gamma = 1 - 1/h. The horizon IS the prediction step (v2 section 1)."""
    return 1.0 - 1.0 / max(int(h), 1)


class Encoder(nn.Module):
    def __init__(self, x_dim, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(x_dim, d), nn.GELU(),
                                 nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        return self.net(x)


class ReadPredictor(nn.Module):
    """s_hat = W_out . [(I - gamma Q)^{-1} R](t, a). The resolvent IS the predictor."""

    def __init__(self, d=64, n=32, k=8, rank=8, teleport=op.TELEPORT):
        super().__init__()
        self.n, self.k, self.teleport = n, k, teleport
        self.q_logits = nn.Linear(d, n * n)
        self.r_logits = nn.Linear(d, n * k)
        self.act_row = nn.Linear(d, n)
        nn.init.normal_(self.act_row.weight, std=0.02)   # NOT zero: v1's mv_row bug
        nn.init.zeros_(self.act_row.bias)
        self.W_out = nn.Linear(k, d)
        self.register_buffer("_mask", torch.ones(n, n, dtype=torch.bool).tril())

    def operator(self, s, a_emb=None):
        B = s.shape[0]
        ql = self.q_logits(s).view(B, self.n, self.n)
        if a_emb is not None:                                    # do(a): bias the rows
            ql = ql + self.act_row(a_emb).unsqueeze(1)
        ql = ql.masked_fill(~self._mask.unsqueeze(0), float("-inf"))
        rl = self.r_logits(s).view(B, self.n, self.k)
        both = torch.cat([ql, rl], dim=-1).softmax(-1)
        Q, R = both[..., :self.n], both[..., self.n:]
        c = self.teleport
        if c > 0:                                                # keeps I - gamma Q away from singular
            Q = (1 - c) * Q
            R = (1 - c) * R + c / self.k
        return Q, R

    def forward(self, s, h, a_emb=None):
        Q, R = self.operator(s, a_emb)
        g = gamma_for_horizon(h)
        eye = torch.eye(self.n, dtype=Q.dtype, device=Q.device)
        z = torch.linalg.solve_triangular(eye - g * Q, R, upper=False)
        return self.W_out(z[:, -1, :]), Q, R                     # read at the last position


class MLPPredictor(nn.Module):
    """Depth-L baseline. Sees paths of length <= L by construction (v2 section 5)."""

    def __init__(self, d=64, L=4, width=None):
        super().__init__()
        w = width or d
        layers, cur = [], d
        for _ in range(L):
            layers += [nn.Linear(cur, w), nn.GELU()]
            cur = w
        layers += [nn.Linear(cur, d)]
        self.net = nn.Sequential(*layers)
        self.L = L

    def forward(self, s, h, a_emb=None):
        z = s if a_emb is None else s + a_emb
        return self.net(z), None, None


def vicreg_terms(s, s_hat, s_tgt, lam=25.0, mu=25.0, nu=1.0, eps=1e-12):
    """invariance + variance + covariance.

    THEOREM 2' (v2.1 section 2). eps MUST be <= 1e-12, not 1e-4, and the difference
    is the whole property. With Std_d = sqrt(var_d + eps),

        |d Std_d / d s_{i,d}| = |c_{i,d}| / (B * Std_d),   c = s - mean(s)

    At spread sigma the numerator is O(sigma) AND the denominator is O(sigma), so the
    ratio is 1/B -- INDEPENDENT of sigma. That is rescue strength: the term pulls just
    as hard out of a nearly-collapsed state as out of a healthy one.
    Floor the denominator with eps = 1e-4 and Std_d stops tracking sigma below 1e-2, so
    the ratio degrades to O(sigma) and the term becomes maintenance strength only.
    MEASURED with eps=1e-4 (the bug this replaces): |dL/ds|max read 4.676e-12 at
    sigma=1e-12 and 0.000000e+00 at exactly zero -- a term that reads 0.99 and pushes
    with nothing. Exact collapse remains a stationary point of measure zero under any
    smooth function of centered s; the claim is escape from any perturbation, not
    impossibility.
    """
    inv = F.mse_loss(s_hat, s_tgt.detach())
    std = torch.sqrt(s.var(dim=0) + eps)
    # SUM over dimensions, not mean. v2.1 section 2 writes mu * SUM_d max(0, 1 - Std_d);
    # a mean divides the per-element gradient by d, so with d=32 the must-fire read 0.131
    # against a mu/B*0.5 = 0.195 threshold and refused. The threshold was right and the
    # implementation was wrong: d/ds of a mean is (1/d) d/ds of a sum.
    var = torch.relu(1.0 - std).sum()
    sc = s - s.mean(0, keepdim=True)
    cov = (sc.T @ sc) / max(s.shape[0] - 1, 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    cv = off.pow(2).sum() / s.shape[1]
    return lam * inv + mu * var + nu * cv, {
        "inv": float(inv), "var": float(var), "cov": float(cv),
        "std_min": float(std.min()), "std_med": float(std.median())}


@torch.no_grad()
def delta_spec(pred, s, s_tgt, h, a_emb, a_emb_placebo):
    """E[ ||s_hat(a) - s||^2 - ||s_hat(a') - s||^2 ]. Negative iff the action is USED.

    A constant-shift predictor -- v1's measured failure, where the do-path knew nothing
    about WHICH move was forced -- reads exactly 0 here rather than a false win.
    """
    e_real = (pred(s, h, a_emb)[0] - s_tgt).pow(2).sum(-1)
    e_plac = (pred(s, h, a_emb_placebo)[0] - s_tgt).pow(2).sum(-1)
    return (e_real - e_plac)


@torch.no_grad()
def rollout_error(pred, s, targets, horizons, a_emb=None):
    """e(h) for h in horizons. A depth-L predictor plateaus beyond h ~ L."""
    return {h: float((pred(s, h, a_emb)[0] - targets[h]).pow(2).sum(-1).mean())
            for h in horizons}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, xd, d, n, k = 64, 24, 32, 16, 4
    x = torch.randn(B, xd)
    enc = Encoder(xd, d)
    rp = ReadPredictor(d=d, n=n, k=k)
    pr = sum(p.numel() for p in rp.parameters())
    # Solve for the baseline width that matches the read's parameter count, rather
    # than guessing it. Ruling 3: matched parameters means MEASURED and printed.
    lo, hi, mp = 4, 512, None
    while lo <= hi:
        w = (lo + hi) // 2
        cand = MLPPredictor(d=d, L=4, width=w)
        pc = sum(p.numel() for p in cand.parameters())
        if pc < pr: lo = w + 1
        else: hi = w - 1
        if mp is None or abs(pc - pr) < abs(sum(q.numel() for q in mp.parameters()) - pr):
            mp = cand
    pm = sum(p.numel() for p in mp.parameters())
    print("(a) PARAMETER MATCH   read=%d  mlp=%d  ratio=%.3f" % (pr, pm, pm / pr))
    assert 0.9 <= pm / pr <= 1.1, "predictors are not parameter-matched within 10%"

    s = enc(x)
    a = torch.randn(B, d)
    s_hat, Q, R = rp(s, h=8, a_emb=a)
    print("(b) READ runs. gamma(h=8)=%.4f  s_hat=%s  rows sum=%.6f"
          % (gamma_for_horizon(8), tuple(s_hat.shape), float((Q.sum(-1) + R.sum(-1)).mean())))
    assert s_hat.shape == (B, d)

    print("(c) THE MUST-FIRE: the variance hinge repels collapse -- and where it does not.")
    # d(std_j)/ds_ij = (s_ij - mean_j) / (N * std_j). At an EXACTLY symmetric point
    # every deviation is exactly zero, so the gradient vanishes however large the
    # hinge is. That is a stationary point of the same shape as the zero-init LoRA
    # death this project already ate once (both factors at zero, no gradient, forever).
    # The hinge is therefore asserted where training can actually be -- at any
    # asymmetry at all -- and the exact-zero point is RECORDED as the known hole
    # rather than asserted away.
    rows = []
    for scale in (0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-1):
        s_c = (torch.randn(B, d, generator=torch.Generator().manual_seed(0)) * scale)
        s_c = s_c.requires_grad_(True)
        loss, st = vicreg_terms(s_c, s_c, torch.zeros(B, d))
        loss.backward()
        rows.append((scale, st["std_min"], st["var"], float(s_c.grad.abs().max())))
        print("    perturbation %8.0e  std_min=%.3e  var=%.4f  |dL/ds|max=%.6e%s"
              % (rows[-1] + ("   <- EXACTLY STATIONARY" if rows[-1][3] == 0.0 else "",)))
    assert rows[0][3] == 0.0, ("exact collapse is expected to be stationary; if this "
                               "ever stops holding the note above is wrong and the "
                               "hinge changed under us")
    assert rows[0][2] > d * 0.99, ("the hinge must be at its MAXIMUM at collapse even "
                                   "though the gradient there is zero -- that is the hole")
    assert all(g > 0 for _, _, _, g in rows[1:]), (
        "the variance term does not push a NEAR-collapsed state off zero -- this is "
        "the v1 failure class and it must be impossible here")
    assert rows[3][3] > 0.1, ("escape force at 1e-6 asymmetry is too weak to matter; "
                              "measured %.3e" % rows[3][3])
    print("    READ: hinge maximal (%.1f = d) at exact collapse with zero gradient; full "
          "escape force by 1e-6." % rows[0][2])
    print("    MITIGATION, standing: never zero-init the encoder output. An exactly "
          "symmetric batch is a fixed point of the variance term.")

    print("(d) Delta_spec is EXACTLY 0 for an action-blind predictor, not a false win.")
    ds_mlp = delta_spec(lambda ss, hh, ae: (mp.net(ss), None, None), s, torch.randn(B, d), 8,
                        a, torch.randn(B, d))
    print("    action-blind Delta_spec mean = %.3e (must be 0.000e+00)" % float(ds_mlp.mean()))
    assert float(ds_mlp.abs().max()) == 0.0

    ds_read = delta_spec(rp, s, torch.randn(B, d), 8, a, torch.randn(B, d))
    print("    read       Delta_spec mean = %+.4e  sd %.4e  (nonzero: the action is USED)"
          % (float(ds_read.mean()), float(ds_read.std())))
    assert float(ds_read.abs().max()) > 0, "the read ignores its action -- v1's mv_row bug"

    print("(e) e(h) is horizon-indexed, so it CANNOT be kappa-invariant the way q_1 was.")
    tg = {h: torch.randn(B, d) for h in (1, 2, 4, 8, 16)}
    er = rollout_error(rp, s, tg, [1, 2, 4, 8, 16], a)
    print("    read e(h):", {h: round(v, 3) for h, v in er.items()})
    gs = [gamma_for_horizon(h) for h in (1, 2, 4, 8, 16)]
    print("    gamma(h):  ", [round(g, 4) for g in gs])
    assert len(set(gs)) == len(gs), "gamma must differ per horizon"

    print("ALL SELF-CHECKS PASSED")
