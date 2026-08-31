"""THEORY.md §8 risk 1 and the DEQ family's documented weakness, bound to a run.

Two separate claims are attacked.

A. "DEQ training instability" as an actual event with a step number, not a vibe.
   A small DEQ is trained with the EXACT implicit gradient through (I - J_f)^-1,
   the way THEORY.md's §6 pipeline requires. rho(J_f), solver iterations, cond(I-J)
   and loss are logged every step. The test asserts the run stays healthy.

B. THEORY.md §8 risk 1: "No proof yet that the composition is a contraction."
   There is an exact answer, and it is worse than "unproven". For row-stochastic P
   the resolvent (I - gamma P)^-1 is non-negative with every row summing to
   1/(1-gamma), so its infinity-norm is EXACTLY 1/(1-gamma). An outer DEQ wrapped
   around it therefore needs Lipschitz constant < (1 - gamma) to contract. gamma is
   the discount: gamma -> 1 is the whole point of a successor representation, and it
   drives the outer model's admissible Lipschitz budget to zero.
"""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theory_ref import deq_forward, jacobian_spectral_radius, implicit_grad_matrix, rows_softmax  # noqa: E402

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deq_run.jsonl")


# ============================================================ A. actual divergence

class DEQ(torch.nn.Module):
    """z* = tanh(W z* + U x + b). No weight norm, no layer norm.

    Bai/Koltun/Kolter 2021 (arXiv 2106.14342) Sec 3.3 report exactly this
    configuration diverging: "Without layer normalization at the end ... the DEQ
    quickly diverges after 25K training iterations." THEORY.md specifies no
    normalization on its equilibrium.
    """

    def __init__(self, d=24, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = torch.nn.Parameter(torch.randn(d, d, generator=g, dtype=torch.float64) * 0.10)
        self.U = torch.nn.Parameter(torch.randn(d, d, generator=g, dtype=torch.float64) * 0.10)
        self.b = torch.nn.Parameter(torch.zeros(d, dtype=torch.float64))
        self.d = d

    def f(self, z, x):
        return torch.tanh(z @ self.W.T + x @ self.U.T + self.b)

    def forward(self, x, max_iter=60, tol=1e-9):
        with torch.no_grad():
            z, n_it, conv, res = deq_forward(lambda z: self.f(z, x),
                                             torch.zeros(x.shape[0], self.d, dtype=torch.float64),
                                             max_iter=max_iter, tol=tol)
        z = z.detach().requires_grad_()
        z_next = self.f(z, x)          # one differentiable step at the fixed point
        return z_next, n_it, conv, res


def _implicit_backward(model, z_star, z_next, grad_out):
    """Exact implicit differentiation: v^T = g^T (I - J)^-1, then backprop v through f.

    This is the (I - J_f)^-1 of THEORY.md §9 / Bai-Kolter-Koltun 2019.
    """
    J = torch.autograd.grad(z_next, z_star, torch.eye(z_star.numel(), dtype=torch.float64)
                            .reshape(-1, *z_star.shape), is_grads_batched=True,
                            retain_graph=True)[0].reshape(z_star.numel(), z_star.numel())
    A = torch.eye(J.shape[0], dtype=torch.float64) - J
    cond = float(torch.linalg.cond(A))
    rho = float(torch.linalg.eigvals(J).abs().max())
    v = torch.linalg.solve(A.T, grad_out.reshape(-1)).reshape(z_star.shape)
    torch.autograd.backward(z_next, v, retain_graph=False)
    return rho, cond


def run_deq_training(steps=4000, lr=1e-2, d=24, seed=0, nfe_budget=30, tol=1e-6, log_path=LOG):
    torch.manual_seed(seed)
    model = DEQ(d=d, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(16, d, generator=g, dtype=torch.float64)
    # Target: the fixed point of a teacher whose own contraction is marginal (rho ~= 0.95).
    Wt = torch.randn(d, d, generator=g, dtype=torch.float64)
    Wt *= 0.95 / torch.linalg.eigvals(Wt).abs().max().real
    y, _, _, _ = deq_forward(lambda z: torch.tanh(z @ Wt.T + x), torch.zeros(16, d, dtype=torch.float64),
                             max_iter=500, tol=1e-12)
    y = y * 3.0  # outside tanh's range: forces the student to grow ||W||

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rows = []
    for step in range(steps):
        with torch.no_grad():
            z_fp, n_it, conv, res = deq_forward(lambda z: model.f(z, x),
                                                torch.zeros(16, d, dtype=torch.float64),
                                                max_iter=nfe_budget, tol=tol)
        z_leaf = z_fp.detach().requires_grad_()
        z_out = model.f(z_leaf, x)
        loss = ((z_out - y) ** 2).mean()
        go = torch.autograd.grad(loss, z_out, retain_graph=True)[0]
        opt.zero_grad()
        rho, cond = _implicit_backward(model, z_leaf, z_out, go)
        opt.step()

        rows.append(dict(step=step, loss=float(loss), rho=rho, cond=cond,
                         iters=n_it, converged=bool(conv), residual=res,
                         Wnorm=float(model.W.norm())))
        if not torch.isfinite(loss) or rho >= 1.0:
            break   # keep going through solver failure: the point is that loss hides it
    with open(log_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows


def test_deq_solver_keeps_converging_and_loss_curve_reflects_it():
    """A 1B run must not lose its fixed point, and if it does, the loss must say so.

    Measured (d=24, lr=1e-3, 30-NFE budget, exact implicit gradient, seed 0):
      step   0  loss 4.4155  rho 0.4942  cond 3.75e0  NFE 21  converged
      step  47  SOLVER STOPS CONVERGING -- no exception, no NaN
      step 480  loss 1.5214  rho 0.8114  cond 1.29e1  residual 3.4e-4
      step 969  loss 1.4568  rho 1.0086  cond 3.32e2  residual 2.2e-2

    The loss decreases monotonically across all 922 steps after the solver failed.
    rho(J_f) crossing 1 means the fixed point no longer exists at all.
    """
    rows = run_deq_training(steps=4000, lr=1e-3)
    first_fail = next((r for r in rows if not r["converged"]), None)
    assert first_fail is None, (
        f"inner solver blew its {30}-NFE budget at step {first_fail['step']} "
        f"(residual {first_fail['residual']:.2e}, rho={first_fail['rho']:.4f}) while "
        f"loss was still {first_fail['loss']:.4f} and falling. "
        f"Run continued {len(rows) - first_fail['step']} more steps; final "
        f"rho={rows[-1]['rho']:.4f}, cond(I-J)={rows[-1]['cond']:.3e}, "
        f"residual={rows[-1]['residual']:.2e}, loss={rows[-1]['loss']:.4f}. "
        f"Nothing in the loss curve marks the failure."
    )


def test_loss_curve_is_a_usable_alarm_for_equilibrium_loss():
    """The rollback question: can you detect this from what a training job normally logs?"""
    rows = run_deq_training(steps=4000, lr=1e-3)
    fail = next((r for r in rows if not r["converged"]), None)
    assert fail is not None, "expected the solver failure reproduced above"
    after = [r for r in rows if r["step"] > fail["step"]]
    worse = [r for r in after if r["loss"] > fail["loss"]]
    assert worse, (
        f"loss never once rose after the fixed point was lost at step {fail['step']}: "
        f"{fail['loss']:.4f} -> {rows[-1]['loss']:.4f} over {len(after)} steps. "
        "A loss-only monitor cannot detect equilibrium loss. rho(J_f) and the "
        "fixed-point residual must be logged explicitly; THEORY.md specifies neither."
    )


# ================================================= B. nested fixed point, exactly

def test_resolvent_infinity_norm_is_exactly_one_over_one_minus_gamma():
    """The amplification factor the outer DEQ must overcome. Not a bound; an identity."""
    torch.manual_seed(0)
    for gamma in (0.5, 0.9, 0.99, 0.999):
        P = rows_softmax(torch.randn(64, 64, dtype=torch.float64))
        M = torch.linalg.inv(torch.eye(64, dtype=torch.float64) - gamma * P)
        assert (M >= -1e-12).all(), "resolvent must be non-negative for stochastic P"
        torch.testing.assert_close(M.sum(1), torch.full((64,), 1 / (1 - gamma), dtype=torch.float64))


@pytest.mark.parametrize("gamma", [0.9, 0.99, 0.999])
def test_outer_deq_still_contracts_when_gamma_grows(gamma):
    """THEORY.md §6 stacks a DEQ on top of the resolvent solve; §8 risk 1 calls the
    composition's contraction "unproven". It is not unproven -- it is a measured
    function of gamma, and gamma is the one knob the successor reframe exists to expose.

    Outer map held FIXED at gain 0.5 with ||C||_inf = 1. Only gamma varies.
    Measured rho of the composed map (d=48, seed 1):
        gamma 0.5   -> 0.081      gamma 0.98  -> 0.286
        gamma 0.8   -> 0.098      gamma 0.99  -> 0.405
        gamma 0.9   -> 0.130      gamma 0.995 -> 0.573
        gamma 0.95  -> 0.182      gamma 0.999 -> 1.280   DIVERGES
    """
    torch.manual_seed(1)
    d = 48
    P = rows_softmax(torch.randn(d, d, dtype=torch.float64))
    M = torch.linalg.inv(torch.eye(d, dtype=torch.float64) - gamma * P)
    C = torch.randn(d, d, dtype=torch.float64)
    C = C / C.abs().sum(1, keepdim=True).max()   # ||C||_inf = 1
    G = 0.5 * (M @ C)
    rho_outer = float(torch.linalg.eigvals(G).abs().max())

    z, n_it, conv, res = deq_forward(lambda z: G @ z, torch.randn(d, dtype=torch.float64),
                                     max_iter=200, tol=1e-10)
    assert rho_outer < 1.0 and conv, (
        f"outer DEQ diverges at gamma={gamma}: rho(outer)={rho_outer:.3f}, "
        f"converged={conv}, residual={res:.3e}. The outer layer never changed; only "
        f"the discount did. ||(I-gamma P)^-1||_inf = {1 / (1 - gamma):.0f}."
    )


@pytest.mark.parametrize("gamma", [0.9, 0.99, 0.999])
def test_saturating_outer_deq_equilibrium_carries_signal(gamma):
    """The other horn: a squashing outer map does not diverge, it goes dead.

    Measured with g(z) = 0.5*tanh((I-gamma P)^-1 C z), d=48, seed 1:
        gamma 0.5/0.9/0.99 -> converges to the TRIVIAL fixed point z=0
                              (median |pre-activation| = 0.00)
        gamma 0.999        -> does not converge in 400 iters, 100% of units
                              saturated, median tanh'(a) = 3.9e-06
    Either way the equilibrium carries no signal. This is the same shape as
    THEORY.md §8 risk 2 ("ker Delta_F may be trivial"), arriving by a second route.
    """
    torch.manual_seed(1)
    d = 48
    P = rows_softmax(torch.randn(d, d, dtype=torch.float64))
    M = torch.linalg.inv(torch.eye(d, dtype=torch.float64) - gamma * P)
    C = torch.randn(d, d, dtype=torch.float64)
    C = C / C.abs().sum(1, keepdim=True).max()
    seen = {}

    def g(z):
        a = M @ (C @ z)
        seen["a"] = a
        return 0.5 * torch.tanh(a)

    z, n_it, conv, res = deq_forward(g, torch.randn(d, dtype=torch.float64),
                                     max_iter=400, tol=1e-10)
    a = seen["a"]
    deriv = float((1 - torch.tanh(a) ** 2).median())
    assert conv and 1e-3 < deriv < 0.999, (
        f"gamma={gamma}: equilibrium carries no signal. converged={conv}, "
        f"median |pre-activation|={float(a.abs().median()):.2f}, "
        f"median tanh'={deriv:.2e}, ||z*||={float(z.norm()):.2e}. "
        "tanh' ~ 1 with z* ~ 0 is the trivial fixed point; tanh' ~ 0 is saturation."
    )


if __name__ == "__main__":
    rows = run_deq_training()
    last = rows[-1]
    print(f"stopped at step {last['step']}: rho={last['rho']:.4f} conv={last['converged']} "
          f"cond={last['cond']:.3e} loss={last['loss']:.4f} iters={last['iters']}")
