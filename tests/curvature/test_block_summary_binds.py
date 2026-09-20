"""Does `ceq.arm_smprime.block_summary` bind the survivor merge iteration 1
settled on, and does `read_summary` make it beta-free off ONE call?

THE SURVIVOR, restated (`tests/curvature/test_bucket_numerics.py` sections 3-6
bind why the other two candidates die first): `m_b` is the max over the LIVE
set of `log R_ij + s_ij`, a wholly-dead block dropped BEFORE the max rather
than shifted by an extended real, and `R_ij == 0` carried by a boolean mask
-- never by `log 0 = -inf` meeting a later subtraction. `block_summary`
returns `(m, l, o, carry)` per block; `read_summary` merges the block axis at
whatever `beta` the caller wants, off that ONE call.

THE beta FLOOR IS A PRECONDITION, NOT A BUG. `out_i(beta)` carries the factor
`e^{(1-beta)M}`, which overflows float64's `exp` once `|1-beta| * M` EXCEEDS
`709.782712893384`. At `M = 800` (the `_hot_bed` below) that puts the floor at
`beta >= 1 - 709.782712893384/800 = 0.11277160888326998` -- AT the floor the
factor equals the ceiling bit-for-bit and the true value still fits `DBL_MAX`,
so the boundary is inclusive; strictly below it the TRUE VALUE is not
representable in float64, and no merge -- this one included -- can return a
finite answer for a quantity that does not fit the dtype.

RED FIRST, verbatim, `python -m pytest tests/curvature/test_block_summary_binds.py -q`
against `ceq/arm_smprime.py` at HEAD 1d7863f, before `block_summary` or
`read_summary` existed -- `11 failed in 2.33s`, every failure the same one
line, `AttributeError: module 'ceq.arm_smprime' has no attribute
'block_summary'`, raised at the first call site in each test body:

    FAILED tests/curvature/test_block_summary_binds.py::test_e_exact_table_bucketed_matches_dense_relative_to_row_mass
    FAILED tests/curvature/test_block_summary_binds.py::test_e_summ_one_summary_serves_five_betas
    FAILED tests/curvature/test_block_summary_binds.py::test_e_summ_must_fire_dropping_the_shift_breaks_every_beta_but_one
    FAILED tests/curvature/test_block_summary_binds.py::test_e2_a_closed_gate_zeroes_every_earlier_blocks_carry_bitwise
    FAILED tests/curvature/test_block_summary_binds.py::test_e2_a_dead_blocks_value_perturbation_never_reaches_the_post_gate_read
    FAILED tests/curvature/test_block_summary_binds.py::test_the_beta_floor_is_a_precondition_not_a_bug[1.0-True]
    FAILED tests/curvature/test_block_summary_binds.py::test_the_beta_floor_is_a_precondition_not_a_bug[0.5-True]
    FAILED tests/curvature/test_block_summary_binds.py::test_the_beta_floor_is_a_precondition_not_a_bug[0.11278-True]
    FAILED tests/curvature/test_block_summary_binds.py::test_the_beta_floor_is_a_precondition_not_a_bug[0.11277160888326998-False]
    FAILED tests/curvature/test_block_summary_binds.py::test_the_beta_floor_is_a_precondition_not_a_bug[0.05-False]
    FAILED tests/curvature/test_block_summary_binds.py::test_the_beta_floor_is_a_precondition_not_a_bug[0.0-False]
    11 failed in 2.33s

    >       m, l, o, c = A.block_summary(q, k, v, u, th, block=2)
                         ^^^^^^^^^^^^^^^
    E       AttributeError: module 'ceq.arm_smprime' has no attribute 'block_summary'

ROUND TWO RED, verbatim, `python -m pytest tests/curvature/test_block_summary_binds.py -q`
at the SAME `ceq/arm_smprime.py` (unedited this round) -- `2 failed, 9 passed`,
both reds TEST-SIDE, neither in the shift formula: (1) `test_e_exact_...`'s
normalizer was raw `_row_mass` (pre-division) differenced against `out`
(post-division), off by exactly `Z_i^beta` -- see `_rel_to_mass`'s docstring;
(2) `test_the_beta_floor_...`'s `0.11277160888326998` row asserted `finite is
False` at a value that EQUALS the overflow ceiling bit-for-bit rather than
exceeding it, an off-by-one-grid-point in the parametrize table, not in the
module docstring's `EXCEEDS` wording, which was already correct. Fixed both
in the test file; `ceq/arm_smprime.py` is untouched.
"""
import math

import pytest
import torch

from ceq import arm_smprime as A

DT = torch.float64
S = 16
BLOCK_SIZES = (1, 2, 4, 8, 16)
BETAS = (0.25, 0.5, 0.75, 1.0)
FIVE_BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BAR = 1e-12

#: float64's `exp` ceiling, bisected in `tests/curvature/test_bucket_numerics.py`.
LOG_MAX_F64 = 709.782712893384


# ---------------------------------------------------------------------------
# beds
# ---------------------------------------------------------------------------

def _bed(seed, closed_at, s=S, d=4, scale=4.0):
    """A random causal bed, gate closed at exactly one index.

    `scale=4.0` keeps logits in the tens, large enough that the block shift
    `M` is not a triviality the must-fire test could pass by accident, and
    nowhere near float64's overflow edge.
    """
    g = torch.Generator().manual_seed(seed)
    q = scale * torch.randn(s, d, generator=g, dtype=DT)
    k = scale * torch.randn(s, d, generator=g, dtype=DT)
    v = torch.randn(s, 1, generator=g, dtype=DT)
    u = 0.3 + 0.6 * torch.rand(s, generator=g, dtype=DT)
    th = torch.rand(s, generator=g, dtype=DT) * 2 * math.pi
    u[closed_at] = 0.0
    return q, k, v, u, th


#: `closed_at=8` sits on the boundary of every block size in `BLOCK_SIZES`
#: except 16 (one block, so the notion of "seam" collapses -- still a valid
#: bed, just not a seam there); `closed_at=5` is interior to blocks 2, 4 and 8
#: (never a multiple of them) so the two beds are not testing the same seam.
def _seam_bed():
    return _bed(101, closed_at=8)


def _mid_bed():
    return _bed(102, closed_at=5)


def _hot_bed(big=800.0):
    """`tests/curvature/test_bucket_numerics.py::_logit_bed`, unchanged: one
    planted logit of `big` at `(i, j) = (3, 0)`, every other logit `0`."""
    q = torch.tensor([[0.0], [0.0], [0.0], [big]], dtype=DT)
    k = torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=DT)
    v = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=DT)
    u = torch.ones(4, dtype=DT)
    th = torch.zeros(4, dtype=DT)
    return q, k, v, u, th


def _row_mass(q, k, v, u, th):
    """`Σ_j |w_ij| |v_j|`, read off the shipped `numerator`'s own modulus row
    -- not off `|out|`, which can cancel to the instrument's own residue
    (`ceq/arm_smprime.py:143`'s `1.2246e-16`) and make a `1e-12` bar
    impossible for reasons that have nothing to do with the merge."""
    mod = A.numerator(q, k, u, th)[1]
    return (mod * v[:, 0].abs()).sum(-1)


def _dense(q, k, v, u, th, beta):
    return A.readout(q, k, v, u, th, beta=beta).flatten()


def _rel_to_mass(got, want, normalizer):
    """Relative error against a normalizer the caller supplies.

    Callers that assert against `BAR` pass `mass / Z**beta`, NOT raw
    `mass`: `got`/`want` are `out = num/Z^beta`, taken POST-division, while
    `_row_mass` (line 108) reports the mass PRE-division. Raw `mass` is
    therefore off by exactly `Z_i^beta` and inflates the measurement by
    `1/Z_i^beta` on every row where `Z_i < 1` (the mid bed's row 6 has
    `Z = 6.45e-07`). `mass/Z^beta` is the magnitude of the terms actually
    summed into `out`, which keeps `_row_mass`'s anti-cancellation point --
    it is still never `|out|`, which can cancel to the instrument's own
    residue and make `BAR` impossible for reasons that have nothing to do
    with the merge.
    """
    tiny = torch.finfo(DT).tiny
    return ((got - want).abs() / normalizer.clamp_min(tiny)).max().item()


# ---------------------------------------------------------------------------
# E-EXACT
# ---------------------------------------------------------------------------

def test_e_exact_table_bucketed_matches_dense_relative_to_row_mass(capsys):
    """Bucketed == dense at every beta in `BETAS` and every block size in
    `BLOCK_SIZES`, relative to `mass/Z^beta` -- the mass of the terms
    ACTUALLY SUMMED into `out = num/Z^beta`, `Z = numerator(...)[1].sum(-1)`
    -- on both the seam bed and the mid-block bed. Not raw `_row_mass`: see
    `_rel_to_mass`'s docstring for why that normalizer is off by `Z_i^beta`.
    Prints the beta x block-size table for the record, both columns, so the
    raw-mass inflation this round found is on the record too."""
    beds = {"seam": _seam_bed(), "mid": _mid_bed()}
    worst = {}
    for bed_name, (q, k, v, u, th) in beds.items():
        mass = _row_mass(q, k, v, u, th)
        Z = A.numerator(q, k, u, th)[1].sum(-1)
        for beta in BETAS:
            want = _dense(q, k, v, u, th, beta)
            exact_mass = mass / Z ** beta
            for block in BLOCK_SIZES:
                m, l, o, c = A.block_summary(q, k, v, u, th, block=block)
                got = A.read_summary(m, l, o, c, beta).flatten()
                rel_mass = _rel_to_mass(got, want, mass)
                rel_exact = _rel_to_mass(got, want, exact_mass)
                worst[(bed_name, beta, block)] = (rel_mass, rel_exact)
                assert rel_exact <= BAR, (
                    "bed=%s beta=%s block=%d: bucketed misses dense by %.4e "
                    "relative to mass/Z^beta" % (bed_name, beta, block, rel_exact))
    print("\nE-EXACT table (BAR = %.0e; rel-to-mass is the pre-division "
          "normalizer this round retired, rel-to-mass/Z^beta is what BAR is "
          "actually asserted against):" % BAR)
    print("bed      beta   block   rel-to-mass    rel-to-mass/Z^beta")
    for (bed_name, beta, block), (rel_mass, rel_exact) in sorted(worst.items()):
        print("%-8s %.2f   %2d      %.3e      %.3e" %
              (bed_name, beta, block, rel_mass, rel_exact))


# ---------------------------------------------------------------------------
# E-SUMM
# ---------------------------------------------------------------------------

def test_e_summ_one_summary_serves_five_betas():
    """ONE `block_summary` call, read at five betas, must equal five separate
    `readout` calls -- the whole point of splitting `block_summary` from
    `read_summary`."""
    q, k, v, u, th = _seam_bed()
    m, l, o, c = A.block_summary(q, k, v, u, th, block=4)
    mass = _row_mass(q, k, v, u, th)
    for beta in FIVE_BETAS:
        got = A.read_summary(m, l, o, c, beta).flatten()
        want = _dense(q, k, v, u, th, beta)
        rel = _rel_to_mass(got, want, mass)
        assert rel <= BAR, (
            "beta=%s: the one-summary read misses the dense call by %.4e "
            "relative to row mass" % (beta, rel))


def _read_no_shift(m, l, o, carry, beta):
    """PLANTED NEGATIVE: exponent 0 on `M` -- the shift dropped -- `o /
    l**beta` off the SAME per-block merge and nothing else changed. Must
    disagree with dense at every beta != 1 and agree exactly at beta = 1,
    where `e^{(1-1)M} = 1` makes the shift a no-op and dropping it changes
    nothing."""
    alive = carry > 0
    m_used = torch.where(alive, m, torch.full_like(m, A.NEG))
    M = m_used.amax(-1)
    Msafe = torch.where(torch.isfinite(M), M, torch.zeros_like(M))
    sc = torch.exp(m_used - Msafe.unsqueeze(-1))
    l_tot = (l * sc).sum(-1)
    o_tot = (o * sc.unsqueeze(-1).to(o.dtype)).sum(-2)
    zb = (l_tot ** beta).unsqueeze(-1)
    return o_tot / zb


def test_e_summ_must_fire_dropping_the_shift_breaks_every_beta_but_one(capsys):
    """Both halves of the must-fire, asserted: the plant FIRES (rel > 1e-6)
    at every beta in `FIVE_BETAS` except 1.0, and is silent (rel <= BAR) at
    beta = 1.0. A plant that fires everywhere, or nowhere, proves nothing."""
    q, k, v, u, th = _seam_bed()
    m, l, o, c = A.block_summary(q, k, v, u, th, block=4)
    mass = _row_mass(q, k, v, u, th)
    Z = A.numerator(q, k, u, th)[1].sum(-1)
    readings = []
    for beta in FIVE_BETAS:
        got = _read_no_shift(m, l, o, c, beta).flatten()
        want = _dense(q, k, v, u, th, beta)
        rel = _rel_to_mass(got, want, mass / Z ** beta)
        readings.append((beta, rel))
        if beta == 1.0:
            assert rel <= BAR, (
                "beta=1 should still be exact with the shift dropped, got "
                "rel %.4e" % rel)
        else:
            assert rel > 1e-6, (
                "must-fire did NOT fire at beta=%s: rel %.4e" % (beta, rel))
    print("\nmust-fire readings (exponent 0 on M, relative to mass/Z^beta):")
    for beta, rel in readings:
        print("  beta=%.2f  rel=%.6e%s" % (beta, rel, "  <- clean" if beta == 1.0 else ""))


# ---------------------------------------------------------------------------
# E2
# ---------------------------------------------------------------------------

def test_e2_a_closed_gate_zeroes_every_earlier_blocks_carry_bitwise():
    """Gate closed at index 8, block = 4 (blocks `[0:4)`, `[4:8)`, `[8:12)`,
    `[12:16)`): for every post-gate row (`i >= 8`), blocks 0 and 1 sit
    entirely before the closed index, so `R_ij = 0` for every `j` in them --
    `carry`, `l` and `o` must be bitwise `0.0` there, not merely small."""
    k_idx = 8
    q, k, v, u, th = _bed(103, closed_at=k_idx)
    m, l, o, c = A.block_summary(q, k, v, u, th, block=4)
    post, early = slice(k_idx, S), slice(0, 2)
    assert torch.equal(c[post, early], torch.zeros_like(c[post, early])), \
        "carry is not bitwise 0.0 on a block wholly before the closed gate"
    assert torch.equal(l[post, early], torch.zeros_like(l[post, early])), \
        "l is not bitwise 0.0 on a block wholly before the closed gate"
    assert torch.equal(o[post, early, :], torch.zeros_like(o[post, early, :])), \
        "o is not bitwise 0.0 on a block wholly before the closed gate"


def test_e2_a_dead_blocks_value_perturbation_never_reaches_the_post_gate_read(capsys):
    """Gate closed at index 8; replace `v[:8]` with `1e6`. Post-gate rows
    (`i >= 8`) never read those positions at all -- the read must be BITWISE
    identical -- while every pre-gate row (`i < 8`) has its entire causal
    window inside the perturbed range and must move."""
    k_idx = 8
    q, k, v, u, th = _bed(104, closed_at=k_idx)
    v2 = v.clone()
    v2[:k_idx] = 1e6
    beta = 1.0
    out1 = A.read_summary(*A.block_summary(q, k, v, u, th, block=4), beta).flatten()
    out2 = A.read_summary(*A.block_summary(q, k, v2, u, th, block=4), beta).flatten()
    post, pre = slice(k_idx, S), slice(0, k_idx)
    assert torch.equal(out1[post], out2[post]), (
        "a post-gate row moved under a pre-gate-only value perturbation")
    moved = (out1[pre] - out2[pre]).abs()
    assert bool((moved > 0).all()), "every pre-gate row should move; some did not"
    print("\nmoved magnitudes, pre-gate rows 0..%d (post-gate rows: bitwise 0):" % (k_idx - 1))
    for i, mv in enumerate(moved.tolist()):
        print("  row %2d: |moved| = %.6e" % (i, mv))


# ---------------------------------------------------------------------------
# the beta floor, as a precondition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta,finite", [
    (1.0, True), (0.5, True), (0.11278, True),
    (0.11277160888326998, True), (0.05, False), (0.0, False),
])
def test_the_beta_floor_is_a_precondition_not_a_bug(beta, finite):
    """See the module docstring: `out_i(beta)` carries `e^{(1-beta)M}`, so at
    `M = 800` (this bed) the answer is representable in float64 only for
    `beta >= 1 - 709.782712893384/800 = 0.11277160888326998`. AT the floor
    the true value (`1.7976931e+308` at 60 dps) still fits `DBL_MAX`
    (`1.7976931348623157e+308`) -- the merge lands on it to rel `4.26e-14`
    -- so the boundary is inclusive; strictly below it (`0.05`, `0.0`) the
    true value genuinely overflows (`1.1582605e+330` at `beta=0.05`) and
    `nan` is correct, not a merge defect."""
    q, k, v, u, th = _hot_bed()
    m, l, o, c = A.block_summary(q, k, v, u, th, block=2)
    got = A.read_summary(m, l, o, c, beta).flatten()[3]
    floor = 1 - LOG_MAX_F64 / 800.0
    assert (beta >= floor) is finite, "floor is %.17g" % floor
    assert bool(got == got) is finite, "beta=%s gave %r" % (beta, got)


def test_the_hot_bed_is_exact_where_dense_has_no_answer():
    """THE FILE'S HEADLINE. `_hot_bed` -- the 800-nat spread the survivor
    shift exists for -- appeared exactly once before this test, inside
    `test_the_beta_floor_is_a_precondition_not_a_bug`, and the only thing
    ever asserted about its VALUE there was a NaN check. Every value
    assertion elsewhere in this file runs on `seam`/`mid`, where dense works
    fine and the shift is a no-op. This is the one place the merge is bound
    where it actually matters.

    Dense cannot be the oracle at row 3: `q_3.k_0 = 800` overflows float64's
    `exp` (`800 > LOG_MAX_F64`), so `readout`'s own numerator AND its `Z`
    are both `inf` there and the row reads `nan` at EVERY beta below --
    checked directly, not assumed. The closed form is the oracle instead:
    `u = 1`, `theta = 0` make every gate exactly `1`, so row 3's true value
    is `e^{800(1-beta)}(1+9e^{-800})/(1+3e^{-800})^beta`, and `e^{-800}`
    underflows to bitwise `0.0` in float64, reducing it to
    `exp(800*(1-beta))` -- no mpmath dependency, no new numerical primitive.
    `k_j = 0` for every `j >= 1`, so rows 0..2 never see the planted logit
    at all and their true value is the plain running mean
    `sum(v[:i+1]) / (i+1)**beta`.
    """
    q, k, v, u, th = _hot_bed()
    for beta in (1.0, 0.75, 0.5, 0.2, 0.11278):
        dense3 = _dense(q, k, v, u, th, beta)[3]
        assert torch.isnan(dense3).item(), (
            "beta=%s: dense should have no answer at row 3 (exp(800) "
            "overflows), got %r" % (beta, dense3))
        want3 = math.exp(800.0 * (1 - beta))
        for block in (1, 2, 4):
            m, l, o, c = A.block_summary(q, k, v, u, th, block=block)
            got = A.read_summary(m, l, o, c, beta).flatten()
            rel3 = (got[3] - want3).abs().item() / abs(want3)
            assert rel3 <= 1e-13, (
                "beta=%s block=%d: row 3 misses the closed form e^{800(1-"
                "beta)} by %.4e" % (beta, block, rel3))
            for i in range(3):
                want_i = sum(v[:i + 1, 0].tolist()) / (i + 1) ** beta
                rel_i = (got[i] - want_i).abs().item() / abs(want_i)
                assert rel_i <= 1e-14, (
                    "beta=%s block=%d row=%d: misses sum(v[:i+1])/(i+1)**"
                    "beta by %.4e" % (beta, block, i, rel_i))
