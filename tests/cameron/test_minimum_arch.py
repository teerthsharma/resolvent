"""How few of the four things does the win condition actually need?

THEORY.md needs (1) gamma factored out, (2) P row-stochastic, (3) a hierarchical
schedule, (4) sheaf + stratum gating. Section 7's win condition is intervention
generalization -- nothing else.

Decision 3 is measured in test_schedule_ablation.py. This file measures what
decisions 1 and 2 buy and what they cost, on the executed-code interventional
corpus, because section 1 says its load-bearing sentence is the one about
occupancy being contractive even when the dynamics are not, and section 8's
risk 5 says the stochastic constraint may destroy expressiveness.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import interventional as iv

D, M = iv.STATE_DIM, iv.ACT_DIM


# ------------------------------------------------------------------ fitting

def _tensors(c, device):
    return (torch.tensor(c["z"], device=device), torch.tensor(c["a"], device=device),
            torch.tensor(c["z_next"], device=device))


def fit_torch(c, stochastic: bool, device, steps: int = 1500, seed: int = 0):
    """Same model, same optimiser, same budget. The ONLY difference is whether
    the state->state block is an unconstrained A or gamma * row-stochastic P."""
    torch.manual_seed(seed)
    z, a, y = _tensors(c, device)
    S = torch.zeros(D, D, dtype=torch.float64, device=device, requires_grad=True)
    A = torch.zeros(D, D, dtype=torch.float64, device=device, requires_grad=True)
    g = torch.tensor(0.0, dtype=torch.float64, device=device, requires_grad=True)
    Ck = torch.zeros(M, D, D, dtype=torch.float64, device=device, requires_grad=True)
    B = torch.zeros(D, M, dtype=torch.float64, device=device, requires_grad=True)
    c0 = torch.zeros(D, dtype=torch.float64, device=device, requires_grad=True)
    params = [S, g, Ck, B, c0] if stochastic else [A, Ck, B, c0]
    opt = torch.optim.Adam(params, lr=0.05)

    def state_block():
        if stochastic:
            return torch.sigmoid(g) * torch.softmax(S, dim=1)
        return A

    for _ in range(steps):
        opt.zero_grad()
        pred = z @ state_block().T + a @ B.T + c0
        for k in range(M):
            pred = pred + a[:, [k]] * (z @ Ck[k].T)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        return dict(A=state_block().cpu().numpy(), Ck=Ck.cpu().numpy(), B=B.cpu().numpy(),
                    c=c0.cpu().numpy(),
                    gamma=float(torch.sigmoid(g)) if stochastic else None)


def predict(f, z, a):
    pred = z @ f["A"].T + a @ f["B"].T + f["c"]
    for k in range(M):
        pred = pred + a[:, [k]] * (z @ f["Ck"][k].T)
    return pred


def closed_loop(f, a_row):
    """A_a = A + sum_k a_k C_k : the operator iterated under a sustained action."""
    A = f["A"].copy()
    for k in range(M):
        A = A + a_row[k] * f["Ck"][k]
    return A


def rho(A):
    return float(np.max(np.abs(np.linalg.eigvals(A))))


_FITS_CACHE = {}  # keyed by device: module scope can't depend on the function-scoped
                  # `device` fixture, so cache here to still fit once per device.


@pytest.fixture
def fits(device):
    if device not in _FITS_CACHE:
        train = iv.code_corpus(1500, mode="interventional", seed=20, edit_range=(-3, 3))
        test = iv.code_corpus(400, mode="interventional", seed=21, edit_range=(-3, 3))
        ood = iv.code_corpus(400, mode="interventional", seed=22, edit_range=(7, 9))
        _FITS_CACHE[device] = dict(
            train=train, test=test, ood=ood,
            free=fit_torch(train, stochastic=False, device=device),
            stoch=fit_torch(train, stochastic=True, device=device))
    return _FITS_CACHE[device]


# ----------------------------------------------------- section 1's guarantee

def test_unconstrained_operator_gives_no_contraction_guarantee(fits):
    """Section 1: with A unconstrained, rho is a diagnostic you hope for.

    Passes only if every sustained action leaves the closed loop contractive.
    """
    f, ood = fits["free"], fits["ood"]
    worst = max(rho(closed_loop(f, a)) for a in ood["a"])
    assert worst < 1.0, (
        f"unconstrained closed-loop spectral radius reaches {worst:.4f} >= 1 "
        "under an unseen sustained intervention: the resolvent (I - A)^-1 b is "
        "not a limit of the rollout and iterating the learned map diverges"
    )


def test_row_stochastic_gamma_P_is_contractive_by_construction(fits):
    """Section 1's structural claim, checked rather than hoped for."""
    f = fits["stoch"]
    assert 0.0 < f["gamma"] < 1.0
    assert np.all(f["A"] >= -1e-12), "P has negative entries"
    assert np.allclose(f["A"].sum(axis=1), f["gamma"], atol=1e-9), \
        "rows of A do not sum to gamma"
    assert rho(f["A"]) <= f["gamma"] + 1e-9


def test_the_contraction_guarantee_survives_conditioning_on_an_intervention(fits):
    """The gap between what section 1 proves and what section 2 needs.

    Section 1 constrains the state->state block to gamma*P and concludes
    rho = gamma < 1 by construction. But the object an intervention acts on is
    A_a = T0 + sum_k a_k T_k, and the action blocks T_k are left unconstrained.
    The certificate is only claimed for the zero-action operator.
    """
    f = fits["stoch"]
    zero_action = rho(f["A"])
    worst = max(rho(closed_loop(f, a)) for a in fits["ood"]["a"])
    print(f"\nstochastic fit: rho(gamma*P)={zero_action:.4f} (gamma={f['gamma']:.4f}), "
          f"max rho(T0 + sum a_k T_k) over held-out interventions={worst:.4f}")
    assert worst < 1.0, (
        f"rho(gamma*P)={zero_action:.4f} < 1 as constructed, but conditioning on "
        f"a held-out intervention reaches rho={worst:.4f} >= 1. Constraining T0 "
        "alone does not make the action-conditioned operator contractive, so the "
        "Banach certificate does not cover the do()-conditioned solve."
    )


# ------------------------------------------------------------------- risk 5

def test_row_stochastic_P_preserves_intervention_generalization(fits):
    """Risk 5, measured on the win condition rather than on perplexity."""
    e_free = iv.nrmse(predict(fits["free"], fits["ood"]["z"], fits["ood"]["a"]),
                      fits["ood"]["z_next"])
    e_stoch = iv.nrmse(predict(fits["stoch"], fits["ood"]["z"], fits["ood"]["a"]),
                       fits["ood"]["z_next"])
    e_id = iv.nrmse(fits["ood"]["z"], fits["ood"]["z_next"])
    print(f"\nOOD-intervention NRMSE: unconstrained={e_free:.4f} "
          f"row-stochastic={e_stoch:.4f} do-nothing={e_id:.4f}")
    assert e_stoch <= e_free * 1.10, (
        f"row-stochastic P costs {e_stoch / e_free:.2f}x the OOD-intervention "
        f"error of the unconstrained fit ({e_stoch:.4f} vs {e_free:.4f})"
    )


def test_row_stochastic_P_can_represent_a_decrementing_loop(fits):
    """The concrete shape of risk 5.

    `for _ in range(a): acc -= 1` makes the outcome DECREASE in the literal, so
    the state->state coupling from the literal to the outcome is negative. A
    non-negative row-stochastic P is a convex combination and has no negative
    entry to put there.
    """
    f_free, f_stoch = fits["free"], fits["stoch"]
    # z[0] is the literal a/10, z[2] is the executed outcome y/30
    assert f_free["A"][2, 0] < -0.05, \
        f"unconstrained fit did not find the negative coupling: {f_free['A'][2, 0]:.4f}"
    assert f_stoch["A"][2, 0] < -0.05, (
        f"row-stochastic P puts {f_stoch['A'][2, 0]:.4f} on the literal->outcome "
        "coupling; a convex combination cannot be negative, so a decrementing "
        "loop is outside the hypothesis class"
    )
