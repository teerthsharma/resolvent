"""V20 R15 it.11 -- JUPITER -- Q6's planted negative, re-pointed at the object
the report describes.

Supersedes `test_v20_r15_it9_q6.py::test_q6_planted_negative_marginal_W1_is_
permutation_blind`, struck by the Inspector at `V20_R15_IT89_INSPECTOR.md:47-76`
(STRIKE I-1): that test's "oracle" was `torch.randn`, its `sqrt(2)` measured
`1.4060`, and its `W1 == 0.0` was an identity that holds at every seed, so
`manual_seed(4096)` did no work.

Nothing here is a second implementation of the instrument: the draw and the
label come from `scale.negation_scope`, called at the banked eval geometry
(`n=4096, s=64, d=24, d_model=16, t_star=2`).

The negative is built so it CAN FAIL. Permutation-blindness alone is a
tautology of the sorted-difference formula; the falsifiable claim is the RANK
INVERSION -- W1 ranks the permuted oracle strictly BETTER than a predictor
NRMSE ranks an order of magnitude better -- and that claim depends on the
oracle's actual values, not on the formula.
"""
import math
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scale.negation_scope as _ns  # noqa: E402
from scale.negation_scope import equilibrium_oracle, make_equilibrium_batch  # noqa: E402

#: PROVENANCE GUARD. `kaggle/snapshot/repo/scale/negation_scope.py` is a second
#: copy of this module in the tree. Every number below is a property of the LIVE
#: copy; if an import order ever resolves `scale` to the snapshot the numbers
#: move and the failure must say WHY, not just that a float moved.
assert pathlib.Path(_ns.__file__).resolve() == (ROOT / "scale" / "negation_scope.py").resolve(), _ns.__file__

#: The banked eval geometry: `s=64, d=24, d_model=16, t_star=2`, `n_eval=4096`,
#: `instrument_hash 5d41a63d...9a309` on 40 of 40 cells.
N, S, D, T_STAR, SEED = 4096, 64, 24, 2, 4096

#: `[RUN]` this file, CPU, `python -m pytest tests/jupiter/test_v20_r15_it11_q6_oracle.py`.
#: Re-run of the Inspector's `V20_R15_IT89_INSPECTOR.md:69` figure, bitwise.
NRMSE_PERMUTED = 1.421901019003236


def _oracle():
    x, _y, head, p = make_equilibrium_batch(
        N, S, D, t_star=T_STAR, seed=SEED, device="cpu")
    return equilibrium_oracle(x, head, p)


def _w1(pred, z):
    """1-Wasserstein between the two POOLED empirical marginals: for equal
    sample sizes, the mean absolute difference of the order statistics."""
    return float((torch.sort(pred).values - torch.sort(z).values).abs().mean())


def _nrmse(pred, z):
    return float(((pred - z) ** 2).mean().sqrt()) / float(z.std(unbiased=False))


def test_q6_the_oracle_is_a_point_prediction_and_has_no_state_axis():
    """The F4's first half, on the shipped object rather than on prose."""
    z = _oracle()
    assert z.shape == (N,), z.shape
    assert z.dtype is torch.float32
    assert z.dim() == 1                      # no state axis to put a measure on


def test_q6_marginal_W1_is_permutation_blind_on_the_real_oracle():
    """STRIKE I-1's repair. Same conclusion, on the object the report names."""
    z = _oracle()
    g = torch.Generator().manual_seed(SEED)
    pred = z[torch.randperm(N, generator=g)]

    assert _w1(pred, z) == 0.0                      # exact, not approx
    # Tolerance `1e-9`, not bitwise: `.mean()` is a parallel float32 reduction
    # whose tree depends on the intra-op thread count. The gap to `sqrt(2)` is
    # `7.687e-03`, six orders above that floor, so the strike survives it.
    assert _nrmse(pred, z) == pytest.approx(NRMSE_PERMUTED, rel=1e-9)
    assert _nrmse(pred, z) > 1.0                    # worse than predict-the-mean

    # And the struck claim is struck: the measured value is NOT sqrt(2).
    assert NRMSE_PERMUTED != math.sqrt(2.0)
    assert abs(NRMSE_PERMUTED - math.sqrt(2.0)) > 1e-3


def test_q6_marginal_W1_inverts_the_ranking_against_nrmse():
    """THE NEGATIVE THAT CAN FAIL.

    Two predictors. A returns the oracle's own values in the wrong order; B
    returns the oracle plus small noise. W1 says A is perfect and B is not.
    NRMSE says B beats A by an order of magnitude. The metric and the bed
    disagree about which predictor is better -- that is the disqualification,
    and it is a property of these values, not of the formula.

    It fails if the oracle's marginal degenerates (a constant label makes the
    permutation lossless and NRMSE(A) collapses to 0), if W1 stops separating B
    from the oracle, or if the two orderings ever agree.
    """
    z = _oracle()
    g = torch.Generator().manual_seed(SEED)
    a = z[torch.randperm(N, generator=g)]                        # permuted
    b = z + 0.1 * float(z.std(unbiased=False)) * torch.randn(N, generator=g)

    w1_a, w1_b = _w1(a, z), _w1(b, z)
    nr_a, nr_b = _nrmse(a, z), _nrmse(b, z)

    assert w1_a == 0.0 and w1_b > 0.0                # W1 prefers A, strictly
    assert w1_a < w1_b
    assert nr_b < nr_a / 10.0                        # NRMSE prefers B, by >10x
    assert (w1_a < w1_b) != (nr_a < nr_b)            # the orderings are opposed


def test_q6_marginal_W1_is_not_degenerate_the_control():
    """The control the struck test never had: W1 is NOT identically 0. It
    separates a predictor whose MARGINAL differs, so the 0.0 above is a
    statement about the pairing, not about the scorer being broken."""
    z = _oracle()
    assert _w1(2.0 * z, z) > 0.0
    assert _w1(z + 1.0, z) == pytest.approx(1.0, rel=1e-4)


# ------------------------------------------- Q1 / W1: the domain fact, made runnable
#
# Carried forward at it.11 so an auditor does not read Q1/W1's F0 as resting on a
# domain-empty theorem. `pathProd_eq_Wp` (`lean/CEQ/V16Domain.lean:176`) carries
# `(hm : forall k, 0 < m k)`. `ceq/arm_smprime.py:22-24` asserts in PROSE that BED-M's
# gate is exactly 0 on 126,976 of 131,072 entries and that NO sequence satisfies the
# hypothesis. Grep for `126976` repo-wide returned ZERO hits before this test: the
# number was published and never checked. It is checked here.


def test_q1_bedm_gate_census_no_sequence_satisfies_pathProd_eq_Wp():
    """`ceq/arm_smprime.py:22-24`, re-run. n=2048 (the train draw), s=64, t_star=2."""
    from scale.negation_scope import CH_DRIVE
    x, _y, head, _p = make_equilibrium_batch(
        2048, S, D, t_star=T_STAR, seed=0, device="cpu")
    m = x[:, :, CH_DRIVE]

    assert m.numel() == 131072                       # 2048 rows x 64 positions
    assert int((m == 0).sum()) == 126976             # the published figure, exact
    assert head == 61 and 2048 * (head + 1) == 126976   # and it is structural

    # THE HYPOTHESIS IS SATISFIED BY NOTHING. Not "rarely" -- by zero rows.
    assert int((m.abs() > 0).all(dim=1).sum()) == 0

    # Which is why Q1/W1's F0 is stated through the HYPOTHESIS-FREE pair instead:
    #   pathProd_polar        `lean/CEQ/V16Domain.lean:105`  -- no guard on m
    #   pathProd_eq_zero_iff  `lean/CEQ/V16Domain.lean:129`  -- no guard on m
    # `pathProd_abs` (`:121`) carries the WEAKER `h0 : forall k, 0 <= m k`, which
    # BED-M does satisfy; only `pathProd_eq_Wp`'s strict `0 < m k` is domain-empty.
