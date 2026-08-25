"""Demonstrates `structural_zero.assert_perturbation_moves_output` firing RED
on two REAL, currently-unfixed instruments in this repo -- not a synthetic
example, and not a rebuild of the operator under test.

Both computations below call the shipped functions directly
(`scale.pivot_probe.build_arm`/`select_pivots`, `ceq.eviction`); nothing here
reimplements the tgate operator or the eviction cache. The M2 finding is
already on record in `tests/chase/test_m2_not_in_P_is_structural_zero.py` and
the M4 finding in `tests/chase/test_checklist_kills_are_evaluable.py` -- what
is new here is not the finding, it is that the SAME ~20-line, general-purpose
assertion catches BOTH, where the project's own record shows these were two
separate hand-written investigations, iterations apart, on two different
subsystems (an attention operator and a KV cache). That gap -- one check per
incident instead of one check for the class -- is the tooling failure this
file is arguing against.

Each subsystem gets one calibration test (must be GREEN: the guard can pass)
beside one structural-zero test (RED on purpose: the guard catches the real
defect this repo still ships).
"""
from __future__ import annotations

import torch

from ceq import eviction as ev
from scale.pivot_probe import build_arm, select_pivots
from tests.chase.structural_zero import assert_perturbation_moves_output

# --------------------------------------------------------------------- M2

S, D, K, SEED, N_DRAWS = 32, 16, 8, 0, 8


def _m2_deltas(placement: str) -> list:
    """Mirror of `scale.pivot_probe.run_arm`'s draw loop (same generator order,
    same pivot selection, same `c` pool), keeping the per-draw |delta
    influence| instead of collapsing to a flip rate."""
    g = torch.Generator().manual_seed(SEED)
    dev = torch.device("cpu")
    rnd = lambda *sh: torch.randn(*sh, generator=g).to(dev)
    i, j = S - 1, max(1, S // 4)

    out = []
    for _ in range(N_DRAWS):
        wq, wk, wo = rnd(D, D), rnd(D, D), rnd(D, D)
        x0, v0 = rnd(S, D), rnd(S, D)
        gvec, bet = torch.sigmoid(rnd(S)), torch.sigmoid(rnd(S))

        pivots = select_pivots(x0 @ wk, K, exclude=(i, j))
        pset = set(int(p) for p in pivots)
        pool = ([p for p in pset if p not in (i, j)] if placement == "in_P" else
                [t for t in range(1, S - 1) if t not in pset and t not in (i, j)])
        if not pool:
            continue
        c = pool[int(torch.randint(0, len(pool), (1,), generator=g))]

        vals = []
        for c_val in (rnd(D), rnd(D)):
            x = x0.clone()
            x[c] = c_val
            v = v0.clone().requires_grad_(True)
            qq, kk = x @ wq, x @ wk
            a, hop2 = build_arm("pivot_signed", qq, kk, gvec, bet, pivots,
                                gen=g, device=dev)
            h = (v + a @ v + hop2 @ v) @ wo
            grad, = torch.autograd.grad(h[i].sum(), v, allow_unused=True)
            vals.append(0.0 if grad is None else float(grad[j].sum()))
        out.append(abs(vals[0] - vals[1]))
    return out


def test_calibration_m2_in_P_control_can_pass():
    """The guard must be able to PASS: c IN P really can move the signed
    influence, so this is GREEN."""
    assert_perturbation_moves_output(
        _m2_deltas("in_P"), what="M2 pivot_signed, c in P")


def test_m2_not_in_P_arm_is_a_structural_zero():
    """RED ON PURPOSE. `scale/pivot_probe.py`'s `not_in_P` control still ships
    this defect: `hop2[i,j]` sums only over `P`, so a `c not in P` draw cannot
    reach it, and the pre-registered "control decays as the mechanism
    requires" kill reads a tautology."""
    assert_perturbation_moves_output(
        _m2_deltas("not_in_P"), what="M2 pivot_signed, c NOT in P")


# --------------------------------------------------------------------- M4

N_TOK, D_FEAT, RHO, KEEP = 96, 32, 0.9, 16


def _m4_context(seed: int):
    """tests/w3/test_w3_eviction.py::context, same generator, same edits."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(N_TOK, D_FEAT, generator=g, dtype=torch.float64)
    x[:, 0] *= 0.25
    causal = int(torch.randint(2, N_TOK, (1,), generator=g))
    x[causal, 0] = 3.0
    return x


def _m4_edit(x, tok, seed):
    g = torch.Generator(device="cpu").manual_seed(seed + 7919)
    d = torch.randn(x.shape[1], generator=g, dtype=x.dtype)
    y = x.clone()
    y[tok] = y[tok] + d / d.norm()
    return y


def _m4_deltas_gated() -> list:
    """Perturb the token the GATE crushes -- the arm the eviction test's own
    companion assertion `g > 0.0` already measures. Positive control: it must
    be able to move `settle_evicted`'s output."""
    out = []
    for s in range(8):
        x = _m4_context(s)
        keep = ev.keep_indices(x, KEEP, RHO)
        # a token that gating crushes but has NOT been excluded from `keep`
        # by the eviction pool -- perturb the lowest-salience KEPT token.
        salience = x[:, 0].abs()
        kept_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        kept_mask[keep] = True
        candidates = torch.nonzero(kept_mask, as_tuple=True)[0]
        tok = int(candidates[torch.argmin(salience[candidates])])
        y = _m4_edit(x, tok, s)
        d = float((ev.settle_evicted(y, keep, RHO)
                   - ev.settle_evicted(x, keep, RHO)).abs().max())
        out.append(d)
    return out


def _m4_deltas_evicted() -> list:
    """Perturb the token `lowest_salience_token(..., exclude=keep)` actually
    picks for eviction -- the quantity `tests/w3/test_w3_eviction.py:96`
    scores as a passing control."""
    out = []
    for s in range(8):
        x = _m4_context(s)
        keep = ev.keep_indices(x, KEEP, RHO)
        crushed = ev.lowest_salience_token(x, RHO, exclude=keep)
        y = _m4_edit(x, crushed, s)
        d = float((ev.settle_evicted(y, keep, RHO)
                   - ev.settle_evicted(x, keep, RHO)).abs().max())
        out.append(d)
    return out


def test_calibration_m4_gated_control_can_pass():
    """The guard must be able to PASS on this subsystem too: perturbing a
    token that stays inside `keep` really does move `settle_evicted`."""
    assert_perturbation_moves_output(
        _m4_deltas_gated(), what="M4 eviction, perturbed token kept")


def test_m4_eviction_control_is_a_structural_zero():
    """RED ON PURPOSE. `ceq/eviction.py`'s `lowest_salience_token(...,
    exclude=keep)` guarantees the perturbed token is OUTSIDE `keep`, and
    `settle_evicted` reads `x[keep]` and nothing else -- the same defect as
    M2's `not_in_P` arm, on a different subsystem, caught by the same
    assertion. `tests/w3/test_w3_eviction.py:96` still asserts this `0.0` as
    a passing measurement."""
    assert_perturbation_moves_output(
        _m4_deltas_evicted(), what="M4 eviction, perturbed token excluded from keep")
