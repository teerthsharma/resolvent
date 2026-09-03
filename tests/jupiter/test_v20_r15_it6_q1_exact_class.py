"""Q1 EXACT CLASS, two wings, tolerance bar `<= 1e-6`, all float64.

WING 1 (`ceq/arm_smprime.py`): `path_product` is checked against `pathProd_polar`'s
own closed form -- an explicit double python loop, not the module's cumprod
route -- and then against `hop_scan`, the shipped planted negative, on a draw
with one gate forced to exactly `0`.

WING 2 (`ceq/arm_pl.py`): `readout` at `q = k = 0`, `s = 0` is checked against
the M6 exp-scan class member its own docstring derives (`normalizer`'s
telescoping argument), computed independently with an explicit python loop.
The planted negative (`sign_flip_gate`) shows the softmax row's convex-hull
property (`Asink_nonneg`, `Asink_row_sum`) forbids it from ever reproducing a
sign-flipping path product.

TEST 2's INDEX SET, MEASURED AND NOT ASSUMED. The straddle set (`j <= 6`,
`i >= 7`, the window `(j, i]` containing the forced zero) is where `G` is
exactly `0` by clause 4 -- confirmed. On that SAME set, `hop_scan` also reads
exactly `0` here (`exp(-inf + i*finite) = 0` in IEEE754, since only `c_i` is
`-inf`ish and `c_j` stays finite for `j <= 6`) -- an artifact of the sentinel,
not a representation of the gate. The module's own docstring names the real
failure region: "every later pair, not only the pairs that straddle it" --
pairs with `j >= 7` too, where BOTH `c_i` and `c_j` are `-inf` and
`-inf - (-inf) = nan`. That is the set this test checks `hop_scan` against,
and it is where `G` stays finite (the window there never contains the zero)
while `hop_scan` is uniformly non-finite.
"""
from __future__ import annotations

import cmath
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ceq import arm_pl as pl        # noqa: E402
from ceq import arm_smprime as smp  # noqa: E402

DT = torch.float64
TOL = 1e-6


# ---------------------------------------------------------------------------
# WING 1 -- ceq/arm_smprime.py
# ---------------------------------------------------------------------------

def test_w1_path_product_matches_the_polar_closed_form_to_1e_6():
    S = 16
    torch.manual_seed(0)
    m = torch.rand(S, dtype=DT)
    theta = torch.randn(S, dtype=DT)

    a = smp.gate(m, theta)
    G = smp.path_product(a)

    m_l, th_l = m.tolist(), theta.tolist()
    closed = torch.zeros(S, S, dtype=torch.complex128)
    for i in range(S):
        for j in range(i + 1):
            prod_m, sum_th = 1.0, 0.0
            for k in range(j + 1, i + 1):        # k in (j, i], independent route
                prod_m *= m_l[k]
                sum_th += th_l[k]
            closed[i, j] = prod_m * cmath.exp(1j * sum_th)

    diff = (G - closed).abs().max().item()
    print(f"test1 max |G - closed_form| = {diff:.6e}")
    assert diff <= TOL


def test_w1_represents_a_zero_gate_exactly_and_the_scan_form_cannot():
    S = 16
    torch.manual_seed(0)
    m = torch.rand(S, dtype=DT)
    theta = torch.randn(S, dtype=DT)
    m = m.clone()
    m[7] = 0.0

    G = smp.path_product(smp.gate(m, theta))
    H = smp.hop_scan(m, theta)

    straddle = torch.zeros(S, S, dtype=torch.bool)  # window (j,i] contains 7
    later = torch.zeros(S, S, dtype=torch.bool)      # j >= 7, window excludes 7
    for i in range(S):
        for j in range(i + 1):
            if j < 7 <= i:
                straddle[i, j] = True
            elif j >= 7:
                later[i, j] = True

    # clause 4: the path product is EXACTLY 0 on every window carrying the zero.
    assert torch.all(G[straddle] == 0)

    # the real failure region: G stays finite (no zero in these windows) while
    # hop_scan's -inf cumsum poisons every one of them to nan.
    g_finite = torch.isfinite(G[later])
    h_nonfinite = ~torch.isfinite(H[later])
    assert torch.all(g_finite)
    assert (G[later] != 0).any()
    assert torch.all(h_nonfinite)

    n_straddle = int(straddle.sum())
    n_differ = int((g_finite & h_nonfinite).sum())
    print(f"test2 straddle entries (window contains index 7), G exactly 0: {n_straddle}")
    print(f"test2 later-pair entries where G is finite but hop_scan is not: {n_differ}")
    assert n_differ == int(later.sum())


# ---------------------------------------------------------------------------
# WING 2 -- ceq/arm_pl.py
# ---------------------------------------------------------------------------

def test_w3_arm_pl_represents_the_exp_scan_class_to_1e_6():
    S = 16
    torch.manual_seed(0)
    g = 0.3 * torch.randn(S, dtype=DT)
    v = torch.randn(S, dtype=DT)
    s = torch.zeros(S, dtype=DT)
    q = torch.zeros(S, 1, dtype=DT)
    k = torch.zeros(S, 1, dtype=DT)

    y = pl.readout(q, k, v, g=g, s=s, diagonal=0)

    C = torch.cumsum(g, dim=-1).tolist()
    v_l = v.tolist()
    y_ref = []
    for i in range(S):
        denom = sum(math.exp(-C[j]) for j in range(i + 1))
        y_ref.append(sum((math.exp(-C[j]) / denom) * v_l[j] for j in range(i + 1)))
    y_ref = torch.tensor(y_ref, dtype=DT)

    diff = (y - y_ref).abs().max().item()
    print(f"test3 max |y - y_ref| = {diff:.6e}")
    assert diff <= TOL


def test_w3_cannot_represent_a_sign_flipping_path_product():
    """PLANTED NEGATIVE `sign_flip_gate`, seed 0."""
    S = 8
    torch.manual_seed(0)
    a = (torch.randint(0, 2, (S,)) * 2 - 1).to(DT)
    b = torch.randn(S, dtype=DT)

    t = torch.zeros(S, dtype=DT)
    for i in range(S):
        prev = t[i - 1] if i > 0 else torch.zeros((), dtype=DT)
        t[i] = a[i] * prev + b[i]

    q = torch.zeros(S, 1, dtype=DT)
    k = torch.zeros(S, 1, dtype=DT)
    ones = torch.ones(S, dtype=DT)
    torch.manual_seed(0)
    n_trials = 200
    for _ in range(n_trials):
        g = 0.5 * torch.randn(S, dtype=DT)
        s = 0.5 * torch.randn(S, dtype=DT)
        A = pl.operator(q, k, g, s, diagonal=0)
        assert torch.all(A >= 0)
        assert torch.allclose(A.sum(-1), ones, atol=1e-12, rtol=0.0)
        y = A @ b
        for i in range(S):
            lo, hi = b[: i + 1].min(), b[: i + 1].max()
            assert y[i] >= lo - 1e-12
            assert y[i] <= hi + 1e-12

    viol = []
    for i in range(S):
        lo, hi = b[: i + 1].min().item(), b[: i + 1].max().item()
        if t[i].item() < lo - 1e-12 or t[i].item() > hi + 1e-12:
            viol.append((i, lo, hi))
    assert len(viol) >= 1
    i0, lo0, hi0 = viol[0]
    print(f"test4 target escapes the hull first at i={i0}: "
          f"t_i={t[i0].item():.6f} not in [{lo0:.6f}, {hi0:.6f}]")
