"""Is one (m, l, o) block summary really beta-free? The triple is. The READOUT
is not, and the window of beta it survives narrows like 1/S.

WHAT LEMMA E1 CLAIMS, in the form it arrived:

    m_b = max_j log|w_ij|,  l_b = sum_j |w_ij| e^{-m_b},  o_b = sum_j w_ij v_j e^{-m_b}
    M = max_b m_b,  l = sum_b l_b e^{m_b - M},  o = sum_b o_b e^{m_b - M}
    out_i(beta) = e^{(1-beta)M} * o / l^beta

WHAT IS ALREADY KNOWN AND MAKES THE BETA-FREEDOM EMPTY. `ceq/arm_smprime.py:255`
is `zb = mod.sum(-1, keepdim=True) ** beta` and `:254` builds `num` with no beta
in it at all. tests/curvature/test_beta_axis_is_a_row_gain.py already turned that
into a kill: beta is a PER-ROW SCALAR GAIN `Z_i^(1-beta)` and nothing else,
measured `worst |softmax_row - linear_row| = 1.110223e-16`. So ANY row summary
that determines `sum_j w_ij v_j` and `Z_i` determines `out_i(beta)` for every
beta, by that result, without E1. E1's only new content is the SHIFT `M`.

WHAT THIS FILE MEASURES ABOUT THAT SHIFT, and it is the hidden beta-dependence.
`M` cancels identically at beta = 1 -- `e^{(1-1)M} = 1` -- and at no other beta.
So the one beta at which E1 is a numerical repair of the shipped head (which
takes NO max at all: `:232` is `e = torch.exp(w.masked_fill(up, NEG))`) is the
one beta at which its shift does not enter the answer. At every other beta the
factor `e^{(1-beta)M}` puts the whole dynamic range back, and the admissible
window is

    |1 - beta| < (log HUGE - log|o / l^beta|) / |M|   ~   709 / |M|   in float64

`M` for this operator is NOT a bounded score max. `|G_ij| = prod_{k=j+1}^{i} m_k`
with `m = clamp(u, 0, 1)` in the CLOSED unit interval, so `log|w_ij|` carries
`sum_k log m_k`, which falls LINEARLY in `i - j`. That is the magnitude half of
the magnitude/phase split, and it is where the beta-dependence hides: the beta
window narrows like `709 / (S |log m|)`, i.e. like `1/S`.

THE CONDITION NOBODY STATED, TWICE.

  (E1) `out_i(beta) = e^{(1-beta)M} o / l^beta` is exact in the reals and
       representable in float only for `|1-beta| < (709 - log|o| + beta log l)/|M|`.
       At beta = 1 the bound is vacuous and E1 is unconditional. At beta = 0 --
       the arm's OTHER shipped corner, `ceq/arm_smprime.py:12-14` -- it is a real
       constraint that the shipped bed violates.

  (E2) `carry_b = 0 ==> every block before it contributes exactly 0` needs
       `M > -inf`, and nothing in E1 or E2 says so. `ceq/arm_smprime.py:244-247`
       is the only place the repo asserts it: "`Z_i` ... is STRICTLY POSITIVE:
       the diagonal term is `R_ii exp(w_ii) = exp(w_ii) > 0` because the empty
       product is `1`." That sentence is FALSE in float64: `exp(w_ii)` is exactly
       `0.0` for `w_ii < -745.2`, reachable with ordinary finite `q` and `k`. At
       `M = -inf` the SAME summary returns three different answers for three
       betas, because `(1-beta)*M` is `0.0 * -inf = nan` at beta = 1 exactly.

THE PLANTED NEGATIVES, which must be SEEN to fire:

  (a) `_e1` must reproduce `arm.readout` to 1e-13 on a TAME bed at every beta.
      A summary that does not is not E1 and every reading below is void.
  (b) a merge that drops the per-block carry magnitude -- `m_b` taken on the
      score alone, ignoring `log prod m` -- must NOT reproduce the row. If it
      does, this file is not testing the carry.
  (c) the shipped head must be SEEN to be non-finite on the hot bed. If it is
      finite there, the bed is not hot and E1 has nothing to repair.

RED FIRST. `test_one_summary_serves_every_beta` and
`test_the_shipped_strictly_positive_diagonal_is_false_in_float64` both failed
against ceq/arm_smprime.py at HEAD 1d7863f before this file's docstring was
written. The verbatim failures are quoted in the FOREMAN report of this round.
"""
import math

import pytest
import torch

import ceq.arm_smprime as arm

SEED = 8801
DK, DV = 4, 3
BLOCK = 4
BETAS = (0.0, 0.5, 1.0, 2.0)
TOL = 1e-13

#: float64's exponent ceiling for `exp`. `math.log(sys.float_info.max)`.
LOG_HUGE = 709.782712893384


# ------------------------------------------------------------------ the beds

def _tame(s=12, seed=SEED):
    """Scores of order 1 and a mild gate: everything is representable here."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(s, DK, generator=g, dtype=arm.DTYPE)
    k = torch.randn(s, DK, generator=g, dtype=arm.DTYPE)
    v = torch.randn(s, DV, generator=g, dtype=arm.DTYPE)
    u = 0.5 + 0.4 * torch.rand(s, generator=g, dtype=arm.DTYPE)
    th = 2 * math.pi * torch.rand(s, generator=g, dtype=arm.DTYPE)
    return q, k, v, u, th


def _hot(s=12, score=800.0, m=0.5, seed=SEED):
    """Every score equal to `score`, every gate magnitude equal to `m`.

    `q = k = c * 1` with `c = sqrt(score * sqrt(DK) / DK)` makes every entry of
    `(q k^T)/sqrt(DK)` exactly `score`, so `M` is `score` (the diagonal's path
    product is the empty product, `1`) and there is no score spread to confound
    the decay with.
    """
    c = math.sqrt(abs(score) * math.sqrt(DK) / DK)
    q = torch.full((s, DK), c, dtype=arm.DTYPE)
    k = torch.full((s, DK), math.copysign(c, score), dtype=arm.DTYPE)
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(s, DV, generator=g, dtype=arm.DTYPE)
    u = torch.full((s,), m, dtype=arm.DTYPE)
    th = torch.zeros(s, dtype=arm.DTYPE)
    return q, k, v, u, th


def _scores(q, k, qk=1.0):
    s = q.shape[-2]
    w = qk * ((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1]))
    up = torch.ones(s, s, dtype=torch.bool).triu(1)
    return w.masked_fill(up, arm.NEG)


# ------------------------------------------------------- E1, from arm's parts

def _summary(q, k, v, u, th, *, block=BLOCK, g=1.0, carry=True):
    """(M, l, o) exactly as LEMMA E1 defines them, built from arm's primitives.

    `carry=False` is planted negative (b): `m_b` taken on the score alone, with
    `log prod m` left out of the max. The merge is otherwise identical.
    """
    m, t2 = arm.blend(u, th, g)
    R = arm.path_product(m)                                # |G| = prod m
    G = arm.path_product(arm.gate(m, t2))                  # complex path product
    s = _scores(q, k)
    logw = torch.log(R) + s                                # log|w_ij|
    vc = v.to(G.dtype)
    n = q.shape[-2]
    mb, lb, ob = [], [], []
    for a in range(0, n, block):
        sl = slice(a, min(a + block, n))
        key = logw[..., sl] if carry else s[..., sl]
        m_b = key.amax(-1)
        dead = torch.isinf(m_b) & (m_b < 0)                # whole block is zero
        safe = torch.where(dead, torch.zeros_like(m_b), m_b)
        l_b = torch.exp(logw[..., sl] - safe.unsqueeze(-1)).sum(-1)
        e = torch.exp(s[..., sl] - safe.unsqueeze(-1))
        o_b = (G[..., sl] * e.to(G.dtype)) @ vc[sl]
        mb.append(torch.where(dead, m_b, safe))
        lb.append(torch.where(dead, torch.zeros_like(l_b), l_b))
        ob.append(torch.where(dead.unsqueeze(-1), torch.zeros_like(o_b), o_b))
    mb = torch.stack(mb, -1)                               # [n, n_blocks]
    M = mb.amax(-1)
    fin = torch.isfinite(M)
    ref = torch.where(fin, M, torch.zeros_like(M))
    sh = torch.exp(mb - ref.unsqueeze(-1))
    l = (torch.stack(lb, -1) * sh).sum(-1)
    o = (torch.stack(ob, -2) * sh.unsqueeze(-1).to(torch.stack(ob, -2).dtype)
         ).sum(-2)
    return M, l, o


def _read(M, l, o, beta):
    """`out_i(beta) = e^{(1-beta)M} o / l^beta`, E1's readout and nothing else."""
    gain = torch.exp((1.0 - beta) * M) / (l ** beta)
    return o * gain.unsqueeze(-1).to(o.dtype)


def _e1(q, k, v, u, th, beta, *, block=BLOCK, carry=True):
    return _read(*_summary(q, k, v, u, th, block=block, carry=carry), beta)


# ---------------------------------------------------- (a) the summary is E1

@pytest.mark.parametrize("beta", BETAS)
def test_e1_reproduces_the_shipped_row_on_a_tame_bed(beta):
    """PLANTED NEGATIVE (a). If this fires, `_summary` is not E1."""
    q, k, v, u, th = _tame()
    got = _e1(q, k, v, u, th, beta)
    want = arm.readout(q, k, v, u, th, beta=beta)
    gap = float((got - want).abs().max())
    scale = float(want.abs().max())
    assert gap <= TOL * max(1.0, scale), (
        "the block summary is not E1 at beta=%.2f: worst |E1 - shipped| = %.6e "
        "against a row scale of %.6e" % (beta, gap, scale))


def test_the_block_max_is_a_gauge_and_cancels_exactly():
    """MEASURED, and it is why `m_b = max_j log|w_ij|` is a CHOICE and not a
    definition: the shift cancels in the merge, so dropping `log|prod m|` out of
    the max moves the tame row by nothing. `m_b` earns its keep only through
    representability, never through the value."""
    q, k, v, u, th = _tame()
    good = _e1(q, k, v, u, th, 1.0, carry=True)
    bad = _e1(q, k, v, u, th, 1.0, carry=False)
    moved = float((good - bad).abs().max())
    assert moved <= TOL, (
        "the block max is not a gauge: dropping log|prod m| moved the row by "
        "%.6e" % moved)


def test_the_o_accumulator_has_no_safe_form_for_a_complex_gate():
    """THE MAGNITUDE/PHASE SPLIT, and it breaks E1 before beta is reached.

    `l_b = sum_j |w_ij| e^{-m_b}` folds the shift INTO its own exponent:
    `exp(log R_ij + s_ij - m_b)`, one argument, `<= 0` by the definition of
    `m_b`, so it is unconditionally safe. `o_b = sum_j w_ij v_j e^{-m_b}` cannot,
    because `w_ij = G_ij e^{s_ij}` and `G_ij` is a PRODUCT, not an exponential.
    The shift has to be applied as `G_ij * exp(s_ij - m_b)`, and
    `s_ij - m_b <= -log R_ij` is POSITIVE and unbounded as the path product
    decays. So `o_b` multiplies an overflowed `exp` by an underflowed `G`.

    The two escapes are both closed by this module's own construction:
      - fold the shift into `log G`: that is `hop_scan` (`:180-193`), the SHIPPED
        planted negative, `nan` on any window carrying a zero magnitude;
      - use the unit phase `G/|G|`: that is `0/0` at a closed gate, the `abs` of
        a complex zero the module refuses at `:150-156`.

    PLANTED NEGATIVE: `l` must stay finite on the same bed. If `l` dies too, the
    bed is simply out of range and this is not a statement about the split."""
    q, k, v, u, th = _hot(s=48, score=0.0, m=1e-12)
    M, l, o = _summary(q, k, v, u, th)
    assert bool(torch.isfinite(l).all()) and bool(torch.isfinite(M).all()), (
        "the magnitude half died too, so the bed is out of range and this test "
        "says nothing: M finite %r, l finite %r"
        % (bool(torch.isfinite(M).all()), bool(torch.isfinite(l).all())))
    bad = int((~torch.isfinite(o)).sum())
    assert bad == 0, (
        "E1's o-accumulator is non-finite in %d of %d entries on a decaying "
        "gate while M and l are both finite: the complex half of the summary "
        "has no safe form, so there IS no (m, l, o) triple to be beta-free "
        "about" % (bad, o.numel()))


def test_the_block_size_does_not_change_the_summarys_answer():
    """One summary, any blocking. E2's merge is associative or it is nothing."""
    q, k, v, u, th = _tame()
    ref = _e1(q, k, v, u, th, 1.0, block=1)
    for b in (2, 3, 4, 6, 12):
        gap = float((_e1(q, k, v, u, th, 1.0, block=b) - ref).abs().max())
        assert gap <= TOL, "block=%d moved the row by %.6e" % (b, gap)


# ------------------------------------- (c) the shipped head has no max at all

def test_the_shipped_head_is_non_finite_on_the_hot_row():
    """PLANTED NEGATIVE (c). `:232` exponentiates the raw score, so a score of
    800 is `inf` before any normalizer exists to cancel it."""
    q, k, v, u, th = _hot()
    out = arm.readout(q, k, v, u, th, beta=1.0)
    assert not bool(torch.isfinite(out).all()), (
        "the shipped head is finite on the hot bed, so the bed is not hot and "
        "E1 has nothing to repair here")


def test_e1_repairs_the_hot_row_at_beta_one():
    """The reroute, measured: at beta = 1 the shift cancels and E1 is exact
    where the shipped head is `nan`. This is the only beta where that holds."""
    q, k, v, u, th = _hot()
    got = _e1(q, k, v, u, th, 1.0)
    assert bool(torch.isfinite(got).all()), "E1 did not repair beta = 1"
    # the beta = 1 row is a probability-weighted mean of v, so it is bounded by
    # max |v| whatever the score scale is.
    assert float(got.abs().max()) <= float(v.abs().max()) + TOL


# ----------------------------------------------- THE RED: is it beta-free?

@pytest.mark.parametrize("beta", BETAS)
def test_one_summary_serves_every_beta(beta):
    """LEMMA E1's own claim: ONE summary, EVERY beta. Read it at each beta from
    the same triple and require a finite answer wherever beta = 1 is finite."""
    q, k, v, u, th = _hot()
    M, l, o = _summary(q, k, v, u, th)
    assert bool(torch.isfinite(_read(M, l, o, 1.0)).all()), "beta=1 is not finite"
    got = _read(M, l, o, beta)
    bad = int((~torch.isfinite(got)).sum())
    assert bad == 0, (
        "one summary does NOT serve beta=%.2f: %d of %d entries are non-finite "
        "while beta=1 from the SAME (M, l, o) is finite. M_max = %.3f, "
        "(1-beta)M = %.3f, float64 admits %.3f"
        % (beta, bad, got.numel(), float(M.max()),
           float((1.0 - beta) * M.max()), LOG_HUGE))


def test_the_shift_belongs_on_the_score_and_not_on_the_weight():
    """THE REPLACEMENT ROUTE, measured. `m_b = max_j log|w_ij|` is the wrong
    shift. Take it on the CONTENT term alone, `m_b = max_j s_ij`, and every term
    of both accumulators is bounded by construction: `exp(s_ij - m_b) <= 1` and
    `|G_ij| <= 1`, so `|o_b| <= sum_j |v_j|` and `l_b <= S` whatever the gate
    does. `Z_i = e^{M_i} l_i` still holds because `l` carries `R_ij` as a FACTOR
    instead of as a logarithm -- which is the same repair `path_product` already
    is against `hop_scan`, applied one level up, to the shift.

    On `m = 1e-12, S = 48`: the log|w| shift leaves 15 of 144 entries of `o`
    non-finite; the score shift leaves 0 of 144, and the readout is finite at
    every beta in BETAS on that bed."""
    q, k, v, u, th = _hot(s=48, score=0.0, m=1e-12)
    M, l, o = _summary(q, k, v, u, th, carry=False)
    assert int((~torch.isfinite(o)).sum()) == 0, "the score shift died too"
    for beta in BETAS:
        r = _read(M, l, o, beta)
        assert bool(torch.isfinite(r).all()), (
            "the score-shifted summary failed at beta=%.2f" % beta)
    # and it is still E1: same value as the shipped head on the tame bed.
    qt, kt, vt, ut, tht = _tame()
    for beta in BETAS:
        got = _e1(qt, kt, vt, ut, tht, beta, carry=False)
        want = arm.readout(qt, kt, vt, ut, tht, beta=beta)
        gap = float((got - want).abs().max())
        assert gap <= TOL * max(1.0, float(want.abs().max())), (
            "the score-shifted summary is not E1 at beta=%.2f: %.6e"
            % (beta, gap))


def test_the_beta_window_is_set_by_the_score_scale_and_not_by_S():
    """RETRACTION, measured. The natural guess is that `|M|` grows with S,
    because `log|w_ij|` carries `sum_k log m_k` and that falls linearly in
    `i - j`. IT DOES NOT. `M_i = max_j (log R_ij + s_ij) >= log R_ii + s_ii =
    s_ii`, because the diagonal's path product is the EMPTY product `1`. So the
    inclusive diagonal FLOORS `M` at the diagonal score and the gate can only
    push `M` down for rows the diagonal does not win -- which is none of them
    when the scores are flat. The admissible `|1-beta| < 709/|M|` window is
    therefore a statement about the SCORE scale alone and is S-invariant."""
    seen = {}
    for s in (12, 24, 48):
        q, k, v, u, th = _hot(s=s, score=0.0, m=0.05)
        M, _, _ = _summary(q, k, v, u, th, block=BLOCK)
        seen[s] = float(M.abs().max())
    assert max(seen.values()) == pytest.approx(0.0, abs=1e-12), (
        "|M| is not floored by the diagonal after all: %r" % seen)
    windows = {}
    for score in (100.0, 400.0):
        q, k, v, u, th = _hot(s=12, score=score, m=0.5)
        M, _, _ = _summary(q, k, v, u, th)
        windows[score] = LOG_HUGE / float(M.abs().max())
    assert windows[100.0] > windows[400.0], (
        "the beta window did not narrow with the score scale: %r" % windows)


def test_the_shipped_strictly_positive_diagonal_is_false_in_float64():
    """`ceq/arm_smprime.py:244-247` asserts `Z_i > 0` because the diagonal is
    `exp(w_ii) > 0`. `exp` underflows to exactly `0.0` below -745.2, so a row
    with every `|w_ij| = 0` is reachable -- and then `(1-beta)*M` is
    `0.0 * -inf = nan` at beta = 1 exactly, and the SAME summary answers three
    betas three different ways."""
    q, k, v, u, th = _hot(score=-800.0, m=0.5)
    _, mod = arm.numerator(q, k, u, th)
    Z = mod.sum(-1)
    assert float(Z.min()) > 0.0, (
        "the shipped head's own guarantee is false: min_i Z_i = %r, so the "
        "STRICTLY POSITIVE claim at ceq/arm_smprime.py:244-247 does not hold "
        "in float64" % float(Z.min()))
