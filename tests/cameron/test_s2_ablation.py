"""S2 — the selection ablation. Kills pre-registered in DONE.md BEFORE any arm ran.

This file exists because CHECKLIST S2 ("pivots+unsigned vs pivots+signed vs
dense+signed") is "required before any claim sentence is published" and the G1
addendum orders it FIRST, before M2 — and `pivot_unsigned` had never been
executed. It binds two things the ablation is worthless without:

  1. the structural theorem, MEASURED, not asserted: a non-negative base
     operator cannot produce a negative influence entry, so its sign-flip rate
     is a floor at exactly 0 and is NOT a flat curve;
  2. that the detector for (1) has been observed FIRING, on the signed arm.
     Nine instruments in this project were internally consistent and externally
     wrong; a non-negativity scan that has never found a negative number is the
     tenth.

It also binds the K4 arm's selector: an "ablated" selector that returns the same
pivots as the real one measures nothing at all.
"""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scale import s2_probe
from scale.pivot_probe import run_arm, select_pivots


def test_random_pivots_are_content_blind_and_not_the_key_norm_set():
    """K4's selector must ignore content AND must actually differ from the
    key-norm selector it ablates."""
    key_a = torch.randn(64, 16, generator=torch.Generator().manual_seed(0))
    key_b = torch.randn(64, 16, generator=torch.Generator().manual_seed(99))

    s2_probe.reseed(7)
    got_a = sorted(int(x) for x in s2_probe.random_pivots(key_a, 8, exclude=(63, 16)))
    s2_probe.reseed(7)
    got_b = sorted(int(x) for x in s2_probe.random_pivots(key_b, 8, exclude=(63, 16)))
    assert got_a == got_b, (
        f"selector read the content: same seed, different keys, different "
        f"pivots {got_a} vs {got_b}")
    assert 63 not in got_a and 16 not in got_a, f"exclude ignored: {got_a}"

    keynorm = sorted(int(x) for x in select_pivots(key_a, 8, exclude=(63, 16)))
    assert got_a != keynorm, (
        f"the random selector returned the key-norm set {keynorm}; the K4 "
        f"ablation would be a no-op dressed as a control")


def test_softmax_pivot_hop2_has_no_negative_influence_entry():
    """The theorem, measured. M = I + A + A[:,P]A[P,:] with A >= 0 entrywise."""
    mn = s2_probe.min_influence_entry("pivot_unsigned", s=32, n_draws=8, k=8)
    assert mn >= 0.0, f"softmax+pivot produced a negative influence entry: {mn!r}"


def test_the_negativity_detector_is_calibrated():
    """A detector never observed FIRING is not a detector. Same scan, signed arm."""
    mn = s2_probe.min_influence_entry("pivot_signed", s=32, n_draws=8, k=8)
    assert mn < 0.0, (
        f"the scan found no negative entry even on the SIGNED arm (min={mn!r}); "
        f"the previous test's pass is vacuous and proves nothing about softmax")


def test_the_share_metric_is_not_itself_degenerate_for_softmax():
    """K3's fallback metric must be finite and NONZERO where `rate` is a floor.
    If term or sigma is also 0 for softmax, K3 is void and there is no fallback."""
    r = run_arm("pivot_unsigned", 32, n_draws=32, k=8, placement="in_P", seed=0)
    assert r["rate"] == 0.0, f"softmax arm flipped a sign: {r!r}"
    assert r["term"] > 0.0 and r["sigma"] > 0.0, (
        f"term/sigma is degenerate for softmax too ({r['term']!r}, "
        f"{r['sigma']!r}); K3 is VOID as a fallback")


def test_absmag_arm_strips_the_sign_and_changes_nothing_else():
    """THE CORRECTED UNSIGNED COMPARATOR.

    The pre-registered unsigned arm is `bench._softmax_operator`, whose scores
    are UNNORMALIZED `q.k/sqrt(d)` while the signed arm's `tgate` scores are
    `qhat.khat/tau` with both vectors L2-normalized. Measured on this box: the
    softmax rows have logit std ~15.8 and a participation ratio of 1.15-1.51,
    i.e. that arm is a hard argmax, not a mixture. Comparing it to tgate on any
    share metric confounds signedness with score scale.

    `absmag` removes the confound completely: it is `|tgate|`, the SAME matrix
    with the sign stripped. Same scores, same tau, same per-row gate, same
    magnitudes entrywise. The only difference from `pivot_signed` is whether the
    background sum can cancel -- which is exactly the axis S2 asks about.

    The bind: `term` must come out BIT-IDENTICAL to the signed arm (the probe
    already takes `.abs()` of c's own term), and the operator must have no
    negative entry. If term differs, the arm changed something besides the sign.
    """
    sgn = run_arm("pivot_signed", 32, n_draws=32, k=8, placement="in_P", seed=0)
    with s2_probe.absmag():
        abs_ = run_arm("pivot_signed", 32, n_draws=32, k=8, placement="in_P", seed=0)
    assert abs_["term"] == sgn["term"], (
        f"absmag changed c's own term {abs_['term']!r} vs {sgn['term']!r}; it is "
        f"not a pure sign ablation")
    assert abs_["n"] == sgn["n"], "absmag changed which draws were used"
    with s2_probe.absmag():
        mn = s2_probe.min_influence_entry("pivot_signed", s=32, n_draws=8, k=8)
    assert mn >= 0.0, f"absmag arm still has a negative entry: {mn!r}"
    assert abs_["sigma"] != sgn["sigma"], (
        "absmag left sigma unchanged; then the background never cancelled in "
        "the signed arm either and the whole small-ball story is empty")


def test_the_gelu_hub_forward_reduces_to_the_probes_own_forward_at_identity():
    """K5-bind. THE NON-DEGENERATE UNSIGNED ARM, and the only thing that makes
    its numbers admissible.

    `pivot_softmax_gelu` cannot go through `run_arm`, because a nonlinearity
    between the hops is not a matrix and `run_arm` builds `h = v + Av + hop2 v`
    from two matrices. So the gelu arms need their own forward -- which is the
    exact hazard this repository has been bitten by five times: a second
    implementation that agrees with the first proves only that two things agree.

    The bind that makes it admissible: with `gelu` replaced by the IDENTITY,

        r = A[P,:] v ;  out = v + Av + A[:,P] r        ==  v + Av + (A[:,P]A[P,:]) v
        m = A v      ;  out = v + Av + A m             ==  v + Av + (A A) v

    are the probe's own two forwards, algebraically. So the reimplementation must
    reproduce `run_arm` BIT-IDENTICALLY in all five fields at identity, on both
    the signed and the softmax operator. If it does not, the draw stream or the
    forward diverged and every gelu number is void.
    """
    for kind in ("pivot_signed", "dense_signed", "pivot_unsigned", "dense_unsigned"):
        want = run_arm(kind, 32, n_draws=24, k=8, placement="in_P", seed=0)
        got = s2_probe.run_hub_arm(kind, 32, n_draws=24, k=8, placement="in_P",
                                   seed=0, nonlin=None)
        for f in ("rate", "k", "n", "term", "sigma"):
            assert got[f] == want[f], (
                f"{kind}: hub forward at identity gives {f}={got[f]!r}, "
                f"run_arm gives {want[f]!r} -- the two forwards are not the same "
                f"computation and no gelu number from this path is admissible")


def test_the_gelu_hub_actually_changes_the_measured_gradient():
    """A nonlinearity that changes no measured number is not being applied.

    NOT compared on `term`/`sigma`: those are computed from the LINEARIZED hop
    weight `A[i,P]*A[P,j]`, which the hub function never touches, so they are
    identical by construction and comparing them would pass on a hub that was
    never wired in. The quantity that must move is the flip count, which comes
    from the autograd path the hub is actually in. Measured on the SIGNED arm,
    where the count is nonzero and therefore has room to move.
    """
    off = s2_probe.run_hub_arm("pivot_signed", 32, n_draws=256, k=8,
                               placement="in_P", seed=0, nonlin=None)
    on = s2_probe.run_hub_arm("pivot_signed", 32, n_draws=256, k=8,
                              placement="in_P", seed=0, nonlin="gelu")
    assert off["k"] > 0, f"the control arm flipped nothing, nothing can move: {off!r}"
    assert on["k"] != off["k"], (
        f"gelu hub gave the identical flip count {on['k']} over {on['n']} draws; "
        f"the hub function is not in the autograd path")


def test_the_matched_softmax_is_a_mixture_and_not_an_argmax():
    """The de-confounded Star-Transformer comparator.

    `bench._softmax_operator` scores UNNORMALIZED `q.k/sqrt(d)`; measured on this
    box its rows have logit std ~15.8 and a participation ratio of 1.15-1.51,
    i.e. it is a hard argmax. `tgate` scores `qhat.khat/tau` with both vectors
    L2-normalized. Comparing them on ANY metric confounds the operator class with
    the score scale, so the honest non-negative comparator is a softmax over the
    SAME normalized scores.

    Binds both halves of that sentence: the matched operator must be a genuine
    mixture (participation ratio well above 1, and rising with s rather than
    pinned near 1), and it must still be entrywise non-negative and row-stochastic
    on the causal support -- otherwise it is not a softmax control at all.
    """
    import torch as _t
    g = _t.Generator().manual_seed(0)
    for s in (32, 512):
        x = _t.randn(s, 16, generator=g)
        wq, wk = _t.randn(16, 16, generator=g), _t.randn(16, 16, generator=g)
        a = s2_probe.matched_softmax_operator(x @ wq, x @ wk, tau=1.0)
        assert float(a.min()) >= 0.0, f"matched softmax has a negative entry at s={s}"
        rows = a[1:].sum(-1)
        assert _t.allclose(rows, _t.ones_like(rows), atol=1e-5), (
            f"matched softmax rows do not sum to 1 at s={s}: {rows.min()}..{rows.max()}")
        pr = float(1.0 / (a[s - 1] ** 2).sum())
        assert pr > 0.25 * s, (
            f"matched softmax at s={s} has participation ratio {pr:.2f} -- it is "
            f"as peaked as the unnormalized one and de-confounds nothing")


def test_the_shipped_operator_swap_really_swaps_the_shipped_operator():
    """THE SCOPE BIND. `tgate` -- the operator every M2 number is measured on --
    lives in `ceq/bench.py` and SHIPS NOWHERE. The Hub config ships `sgate` and
    `ceq/attention.py` ships the row-L1 form, and both carry the denominator that
    `bench.py`'s own docstring names as the measured cause of the 1/s death.

    So "does the property survive on the operator that actually ships" is a
    different question from every number in the M2 journal, and it needs the
    routed hop-2 applied to the SHIPPED matrix. This binds that the swap gives
    exactly `bench._causal_sgate_operator` and not something that merely differs
    from tgate.
    """
    import torch as _t
    from ceq import bench
    g = _t.Generator().manual_seed(0)
    x = _t.randn(32, 16, generator=g)
    wq, wk = _t.randn(16, 16, generator=g), _t.randn(16, 16, generator=g)
    qq, kk = x @ wq, x @ wk
    gv, bt = _t.sigmoid(_t.randn(32, generator=g)), _t.sigmoid(_t.randn(32, generator=g))
    piv = s2_probe.select_pivots(kk, 8)

    base, _ = s2_probe.pivot_probe.build_arm("pivot_signed", qq, kk, gv, bt, piv)
    with s2_probe.shipped():
        got, hop = s2_probe.pivot_probe.build_arm("pivot_signed", qq, kk, gv, bt, piv)
    want = bench._causal_sgate_operator(qq, kk, rho=1.5, lam=0.10)
    assert _t.equal(got, want), "the swap is not bench._causal_sgate_operator"
    assert not _t.equal(got, base), "the swap returned tgate; nothing was swapped"
    assert _t.equal(hop, got[:, piv] @ got[piv, :]), (
        "hop 2 is not routed through the pivots on the swapped operator")
