"""ARMS-DISTINCT for `scale/pivot_probe.py`. G3, reopened in a new file.

WHY THIS EXISTS. The iteration-5 Health Inspector pass found that the pivot probe
defines seven arms and NO distinctness bind covered any of them -- the
iteration-2 bind guards `ceq/bench.py::sign_flip_rate` only. The arms were
structurally distinct at the time of writing, but nothing ASSERTED it, and
"correct today, unasserted" is the state every one of this project's thirteen
broken instruments was in before it broke.

WHY THIS COMPARES TENSORS AND NOT RATES -- the naive bind would have been
instrument fourteen. `dense_unsigned` and `pivot_unsigned` BOTH read exactly
0.0000 as a sign-flip rate, so a rate fingerprint would flag them as duplicates.
They are NOT duplicates. Both are non-negative operators, so the influence
Jacobian `I + A + A^2` is non-negative entrywise and neither can ever flip a
sign -- by the same theorem that makes softmax's floor legitimate rather than
suspicious. Their RATES coincide at a structural zero while their TENSORS differ.
`test_the_rate_level_bind_would_have_false_positived` pins that reasoning so a
future reader does not "simplify" this file back into a wrong one.

WHY IT CALLS `build_arm` RATHER THAN REBUILDING THE OPERATORS. A test that
constructs its own copies proves only that two reimplementations agree. That is
how this project published a parity number of 89,400.180 against 1.667, and how
"alpha=0 is bitwise stock" survived until someone checked it against the real
sdpa path instead of ours (1.490e-07 at 2 kv heads; Qwen2.5-0.5B is GQA).
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scale import pivot_probe as pp


def build_all(s: int = 64, d: int = 16, k: int = 8, seed: int = 0, device=None):
    """Every arm's (A, hop2) from ONE draw, through the probe's own code path."""
    g = torch.Generator().manual_seed(seed)
    dev = device or torch.device("cpu")
    rnd = lambda *sh: torch.randn(*sh, generator=g).to(dev)
    wq, wk = rnd(d, d), rnd(d, d)
    x = rnd(s, d)
    gvec = torch.sigmoid(rnd(s))
    bet = torch.sigmoid(rnd(s))
    qq, kk = x @ wq, x @ wk
    pivots = pp.select_pivots(kk, k)
    out = {}
    for kind in pp.ARMS:
        # each arm gets its own generator so the `random` arm cannot consume
        # draws that would shift a later arm -- stream isolation, same
        # discipline as `gam`/`bet`/`gvec` in ceq/bench.py
        gi = torch.Generator().manual_seed(seed + 1000)
        out[kind] = pp.build_arm(kind, qq, kk, gvec, bet, pivots, gen=gi,
                                 device=dev)
    return out


def collisions(tensors: dict) -> list:
    keys = list(tensors)
    return [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
            if torch.equal(tensors[a], tensors[b])]


# ==========================================================================
# CALIBRATION -- the bind must be seen to FAIL before any green is trusted
# ==========================================================================

def test_the_bind_fires_on_a_known_duplicate():
    """RED-first. One arm under two names must be caught."""
    built = build_all()
    hop2 = {k: v[1] for k, v in built.items()}
    hop2["dense_signed_COPY"] = hop2["dense_signed"]
    got = collisions(hop2)
    assert ("dense_signed", "dense_signed_COPY") in got, (
        f"the bind does not detect an exact duplicate; collisions found: {got}"
    )


def test_the_rate_level_bind_would_have_false_positived():
    """Why this file compares tensors, pinned so nobody 'simplifies' it back.

    Both unsigned arms are non-negative, so `I + A + A^2` is non-negative
    entrywise and neither can flip a sign. Their RATES are therefore both an
    identical structural zero, while their hop-2 TENSORS differ.
    """
    built = build_all()
    a_un, h_un = built["pivot_unsigned"]
    a_de, h_de = built["dense_unsigned"]

    # they share the base operator -- that is BY DESIGN, both are softmax
    assert torch.equal(a_un, a_de), (
        "the unsigned arms no longer share a base operator; this test's premise "
        "has changed and its reasoning must be re-derived"
    )
    # both bases are non-negative, so no sign flip is reachable: rates tie at 0
    assert float(a_un.min()) >= 0.0
    # ...yet the hop-2 tensors differ, which is what the bind must see
    assert not torch.equal(h_un, h_de), (
        "pivot and dense unsigned hop-2 tensors are identical; pivot routing is "
        "not actually restricting anything"
    )


# ==========================================================================
# THE BIND ITSELF
# ==========================================================================

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_no_pivot_probe_arm_reports_another_arms_tensor(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("cuda unavailable")
    built = build_all(device=torch.device(device))
    hop2 = {k: v[1] for k, v in built.items()}
    got = collisions(hop2)
    n = len(pp.ARMS)
    assert not got, (
        f"ARMS-DISTINCT FAILED (G3) in scale/pivot_probe.py: {got}. At least one "
        f"arm is computing another arm's hop-2 operator and would publish that "
        f"number under its own name."
    )
    print(f"\n  pivot-probe ARMS-DISTINCT on {device}: {n} arms, "
          f"{n*(n-1)//2} pairs, 0 collisions")


def numerical_rank(m: torch.Tensor) -> tuple:
    """Rank with an EXPLICIT float32-aware tolerance. Returns (rank, svals).

    `torch.linalg.matrix_rank(m.double())` on a matrix COMPUTED in float32 is
    wrong and was instrument fourteen in this project: casting to double after
    the fact does not recover precision, it promotes float32 rounding noise
    (~1e-8 here) to double and the default tolerance counts it as real rank. On
    the first run that reported rank 37 for a matrix that is provably rank <= 8,
    because (64x8)(8x64) cannot exceed 8.

    Standard tolerance instead: tol = max(dims) * eps(float32) * sigma_max.
    """
    sv = torch.linalg.svdvals(m.double())
    tol = max(m.shape) * torch.finfo(torch.float32).eps * float(sv[0])
    return int((sv > tol).sum()), sv


def test_pivot_routing_actually_restricts_rank():
    """The mechanism, asserted rather than assumed.

    `A[:,P] A[P,:]` has rank <= |P| by construction -- that IS the claim that one
    token's share of the two-hop sum becomes 1/k independent of s. If the pivot
    hop-2 had full rank, the arm would be dense attention wearing a pivot's name.
    """
    k = 8
    built = build_all(s=64, k=k)
    r_piv, sv_piv = numerical_rank(built["pivot_signed"][1])
    r_den, _ = numerical_rank(built["dense_signed"][1])
    assert r_piv <= k, f"pivot hop-2 rank {r_piv} exceeds |P| = {k}"
    assert r_den > k, (
        f"the dense control has rank {r_den} <= |P| = {k}, so this draw cannot "
        f"distinguish routing from density and the comparison is uninformative"
    )
    # the spectral gap is the evidence the tolerance is not doing the work
    gap = float(sv_piv[k - 1]) / max(float(sv_piv[k]), 1e-300)
    assert gap > 1e3, (
        f"no clear spectral gap at |P| = {k} (ratio {gap:.3e}); the rank claim "
        f"would then be an artefact of the tolerance rather than of the routing"
    )
    print(f"\n  rank(pivot hop2) = {r_piv} == k = {k};  rank(dense hop2) = {r_den}"
          f"\n  spectral gap at k: sigma_{k}/sigma_{k+1} = {gap:.3e}")


def test_every_arm_in_ARMS_is_actually_dispatched():
    """An arm accepted by `build_arm` but absent from `ARMS` is unbound; an arm
    in `ARMS` that `build_arm` rejects is a crash waiting for a sweep."""
    built = build_all()
    assert set(built) == set(pp.ARMS)
    g = torch.Generator().manual_seed(0)
    qq = kk = torch.randn(8, 4, generator=g)
    gvec = bet = torch.sigmoid(torch.randn(8, generator=g))
    with pytest.raises(ValueError):
        pp.build_arm("an_arm_nobody_wired_up", qq, kk, gvec, bet,
                     torch.tensor([1, 2]))
