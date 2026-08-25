"""THE PRE-REGISTERED KILL: frustration(A[P,P]) >= 0.30 on the M3 harness data.

WHAT THIS FILE FALSIFIES. The round's composed candidate is: pivot-routed hop 2
(R1, the measured win) over a co-prime dilated band at row width 8 (R5), run at
the lambda that actually leaves sign in the operator (R4), with the batched path
(R6) paying for it. Every one of those pieces is measured. The piece that was
NEVER measured is R4's own open question -- "whether lam=1.0 repairs the
SELECTED set is UNMEASURED" -- and it is the hinge, because if the operator the
routed arm carries has no sign in it, the arm is `pivot_unsigned` with extra
steps and the whole synthesis is a rename.

THE KILL, FIXED BEFORE RUNNING, NEVER TO BE EDITED:

    Zaslavsky frustration of the k x k sub-operator induced on the SELECTED
    pivot set, A[P, P], measured at the M3 harness geometry
    (n=256, s=64, d=24, d_model=16, seed=0, k=8, rho=1.5), must read

        >= 0.30

    Below 0.30 the routed arm is routing a sign-free operator, R2's "sign does
    not beat routing" was never a statement about sign, and this mechanism is
    dead where it stands.

WHY FRUSTRATION AND NOT A NEGATIVE COUNT. Counting negative entries is not
gauge-invariant: switching by D = diag(+-1) sends sign(A_ij) -> d_i d_j
sign(A_ij), which moves the count while changing nothing structural. The
triangle sign product sign(A_ij) sign(A_ik) sign(A_jk) is invariant because each
d appears twice (Harary). `scale/frustration_audit.py` established this and
calibrated it at both ends; this file re-calibrates the SAME statistic under its
own estimator (exhaustive over all C(8,3) triangles of P, not sampled), because
a bar borrowed from a differently-estimated number is a bar that means nothing.

WHY IT IS RED RIGHT NOW, AND THE ONE LINE THAT TURNS IT GREEN. `Arm._operator`
builds `bench._causal_sgate_operator(..., lam=m3.SGATE_LAM)` and SGATE_LAM is
0.10. On THIS task's inputs -- `make_batch` scales x by 0.1, so the logits
q@k.T/sqrt(d) have mean |w| = 2.68e-03 and max 1.47e-01 -- softmax(w) and
softmax(-w) are within a factor exp(2w) ~ 1 of each other everywhere, so
pp - 0.10*pm is POSITIVE at every entry. The arm named `pivot_signed` is
entrywise non-negative on the data it was scored on. The switch is a lam for the
routed arm; this file reads it from `m3.ROUTED_LAM` when that constant exists and
falls back to the shipped `m3.SGATE_LAM` when it does not, so adding the arm is
what flips the colour and nothing here has to be rewritten to claim the pass.
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ceq import bench                                            # noqa: E402
from scale import m3_capability as m3                            # noqa: E402
from scale.negation_scope import make_batch                      # noqa: E402
from scale.pivot_probe import batched_select_pivots              # noqa: E402
from dilated import dilated_mask, sgate_masked                   # noqa: E402

torch.set_num_threads(2)          # a measurement may be running; do not starve it

# ---- the geometry, pinned to the [RUN] that produced softmax 0.877168 --------
N, S, D, DMODEL, KP, SEED = 256, 64, 24, 16, 8, 0

#: THE PRE-REGISTERED NUMBER. Not an inequality chosen after seeing a reading.
BAR = 0.30
#: Instrument law: both ends must be SEEN under THIS estimator or a low reading
#: is not evidence of anything. `frustration_audit.py` reads 0.000000 planted /
#: 0.508600 iid at 20000 sampled triangles; these bracket that pair.
CAL_BALANCED_MAX, CAL_RANDOM_MIN = 0.02, 0.40

COPRIME, WINDOW = (1, 3, 5, 7), 8

#: The routed arm's lambda. Falls back to the shipped constant, which is what
#: makes this file RED against the code as it stands.
ROUTED_LAM = getattr(m3, "ROUTED_LAM", m3.SGATE_LAM)

#: All C(8,3) triangles, each ordered i > j > k so every entry exists in the
#: strictly-lower-triangular causal support. EXHAUSTIVE, so there is no sampling
#: error to report and no seed that can flatter the reading.
TRIANGLES = [tuple(sorted(t, reverse=True))
             for t in itertools.combinations(range(KP), 3)]


def _harness_qk():
    """q, k exactly as `Arm.forward` computes them, on the harness batch."""
    x, _y, _f, _p = make_batch(N, S, D, d_model=DMODEL, seed=SEED)
    torch.manual_seed(SEED)
    arm = m3.Arm("pivot_signed", S)
    with torch.no_grad():
        return arm, x, arm.wq(x), arm.wk(x)


def _sub_block(a: torch.Tensor, piv: torch.Tensor) -> torch.Tensor:
    """A[P, P] per example, with P SORTED so index order is position order."""
    n, s, _ = a.shape
    kp = piv.shape[-1]
    ps, _ = torch.sort(piv, dim=-1)
    rows = torch.gather(a, 1, ps[:, :, None].expand(n, kp, s))
    return torch.gather(rows, 2, ps[:, None, :].expand(n, kp, kp))


def _frustration(sub: torch.Tensor) -> tuple[float, int]:
    """Fraction of triangles with sign product -1. Zero entries are SKIPPED,
    never counted as +1: a masked-out edge is an absent triangle, and folding it
    in as agreement is how a sparse operator reads balanced by construction."""
    sg = torch.sign(sub)
    neg = tot = 0
    for i, j, k in TRIANGLES:
        pr = sg[:, i, j] * sg[:, i, k] * sg[:, j, k]
        tot += int((pr != 0).sum())
        neg += int((pr < 0).sum())
    return (neg / tot if tot else float("nan")), tot


# =========================================================================== #
# 1. CALIBRATION. Runs first. If this misreads, every other number here is void.
# =========================================================================== #
def test_the_frustration_estimator_sees_both_ends():
    g = torch.Generator().manual_seed(SEED)
    d = torch.where(torch.rand(N, S, 1, generator=g) < 0.5, -1.0, 1.0)
    balanced = ((d * d.transpose(1, 2)) * torch.rand(N, S, S, generator=g)).tril(-1)
    iid = torch.randn(N, S, S, generator=g).tril(-1)
    piv = torch.stack([torch.randperm(S, generator=g)[:KP] for _ in range(N)])

    f_bal, n_bal = _frustration(_sub_block(balanced, piv))
    f_iid, n_iid = _frustration(_sub_block(iid, piv))
    assert f_bal <= CAL_BALANCED_MAX, (
        f"planted-balanced reads {f_bal:.6f} over {n_bal} triangles, above "
        f"{CAL_BALANCED_MAX}: the estimator cannot recognise a gauge artifact.")
    assert f_iid >= CAL_RANDOM_MIN, (
        f"iid-random reads {f_iid:.6f} over {n_iid} triangles, below "
        f"{CAL_RANDOM_MIN}: the estimator cannot recognise real sign structure.")


# =========================================================================== #
# 2. THE KILL.
# =========================================================================== #
def test_the_selected_pivot_set_carries_sign():
    arm, x, q, k = _harness_qk()
    with torch.no_grad():
        # BIND, not a reimplementation: at the SHIPPED lam this call is bitwise
        # the tensor `Arm._operator` returns. Instrument #17 was a correct
        # measurement of an operator that ships nowhere.
        shipped = bench._causal_sgate_operator(q, k, rho=m3.SGATE_RHO,
                                               lam=m3.SGATE_LAM, window=0)
        assert torch.equal(shipped, arm._operator(q, k)), (
            "the operator measured here is not the operator the arm builds")

        a = bench._causal_sgate_operator(q, k, rho=m3.SGATE_RHO,
                                         lam=ROUTED_LAM, window=0)
        piv = batched_select_pivots(k, KP)          # top-k by key-norm, UNCHANGED
        f, n_tri = _frustration(_sub_block(a, piv))

        tril = a.tril(-1)
        bulk = float((tril < 0).sum()) / max(1, int((tril != 0).sum()))
        cols = torch.gather(a, 2, piv[:, None, :].expand(N, S, KP))
        sel = float((cols < 0).sum()) / max(1, int((cols != 0).sum()))

    assert f >= BAR, (
        f"KILL: frustration(A[P,P]) = {f:.6f} over {n_tri} triangles, below the "
        f"pre-registered {BAR}. lam={ROUTED_LAM}, bulk negative-entry fraction "
        f"{bulk:.6f}, selected-set negative fraction {sel:.6f}. The routed arm "
        f"is routing a sign-free operator; it is `pivot_unsigned` wearing a name.")


# =========================================================================== #
# 3. The band the mechanism composes over. Structural, bitwise, cheap.
# =========================================================================== #
@pytest.mark.parametrize("dilation", COPRIME)
def test_the_coprime_band_keeps_width_eight_and_a_zero_row_sum(dilation):
    _arm, x, q, k = _harness_qk()
    with torch.no_grad():
        m = dilated_mask(S, x.device, WINDOW, dilation)
        a = sgate_masked(q, k, m, rho=m3.SGATE_RHO, lam=1.00)
        if dilation == 1:
            assert torch.equal(a, bench._causal_sgate_operator(
                q, k, rho=m3.SGATE_RHO, lam=1.00, window=WINDOW)), (
                "dilation 1 is not the shipped windowed operator")
        width = int((a != 0).sum(-1).max())
        rowsum = float(a.sum(-1).abs().max())
    assert width <= WINDOW, (
        f"row width {width} exceeds {WINDOW} at dilation {dilation}: the "
        f"normalizer is summing over more than the band and R5's whole point "
        f"-- reach without an s-sized row -- is gone.")
    assert rowsum < 1e-5, (
        f"row sum |{rowsum:.3e}| at dilation {dilation} is not zero. At lam=1.0 "
        f"softmax(w) and softmax(-w) each sum to 1, so every row must cancel "
        f"EXACTLY; a nonzero sum means the operator is not the difference it "
        f"claims to be.")
