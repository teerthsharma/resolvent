"""Did the market update as a POTENTIAL? A model-free test on quotes alone.

THE CLAIM THIS DECIDES. In a pairwise-comparison market the quoted log-odds are an
EDGE COCHAIN omega on the graph of pairings. Every Elo / Bradley-Terry-class pricer
represents it as a potential -- omega = d0 . r for a rating vector r -- so when
match (i,j) resolves, that class's update to ADJACENT quotes is d0 . dr, and

    d1 . d(omega) = (d1 o d0) . dr = 0

EXACTLY, on every triangle, for any K-factor. VERIFIED: 0.000e+00 across
K in {8, 16, 32, 64} and p in {0.25, 0.50, 0.73}, against 2.5434 for an update
that is not any d0 . dr. demo() reproduces both.

So the curl of the UPDATE is a decision procedure with no model in it: nonzero
curl on the triangles through a resolved pairing means the market's CONSEQUENCE
structure lies outside the range of EVERY potential rating model, and the smallest
class that can predict it is a grade-1 read (ceqjepa/dr1.py).

WHY THIS SHAPE AND NOT THE PREVIOUS ONES. Every separation this project measured
before was on a bed built to exhibit it -- dr1.py's own curl bed generates the
target as d0 @ pot + curl_weight * (d1^T @ circ), so of course a node model cannot
fit it. Here nothing is constructed: the quotes are what a market published, and
the null hypothesis (markets price as potentials) is the one a reader expects to
hold. A NULL RESULT IS THE LIKELY OUTCOME and is a clean retirement, because
bookmakers are themselves mostly rating-class pricers.

THE CONTROL IS THE WHOLE EXPERIMENT. Quotes drift for reasons unrelated to any
resolution -- news, money flow, the clock. So the affected triangles (those through
the resolved pairing) are scored against UNAFFECTED triangles read over the SAME
interval. Without that control this measures market noise and nothing else.

SCOPE, stated because it bounds the claim. The curl-free property is exact for
Elo and Bradley-Terry. Glicko and TrueSkill scale updates by opponent uncertainty
and are NOT exact gradients, so a market pricing with those is outside the null
and a nonzero reading would not convict it. State the control class with any result.

RUN: python -m ceqjepa.market_curl
"""

import numpy as np

from ceqjepa.dr1 import Complex

__all__ = ["pairing_complex", "curl_of_update", "affected_mask", "score_resolution"]


def pairing_complex(n_teams, pairings):
    """The complex whose edges are quoted pairings and whose triangles are the
    triples all three of whose pairings are quoted. A triple missing one quote is
    not a triangle and carries no curl -- it is dropped, not imputed."""
    have = {tuple(sorted(p)) for p in pairings}
    tri = [(a, b, c)
           for a in range(n_teams) for b in range(a + 1, n_teams) for c in range(b + 1, n_teams)
           if {(a, b), (b, c), (a, c)} <= have]
    return Complex(n_teams, tri)


def curl_of_update(cx, omega_before, omega_after):
    """|d1 . (omega_after - omega_before)| per triangle. Zero for any rating-class
    update, whatever its size, by d1 o d0 = 0."""
    return np.abs(cx.d1 @ (np.asarray(omega_after) - np.asarray(omega_before)))


def affected_mask(cx, resolved_edge):
    """Which triangles contain the pairing that resolved.

    WARNING, and it is the reason placebo_interval() below exists. Under ANY
    potential model the unaffected triangles have dr = 0 at all three vertices,
    so their update is identically zero -- both curl AND signal. Comparing
    affected to unaffected therefore tests the NOISE FLOOR, not an alternative
    hypothesis: it cannot tell "the market priced as a potential" from "nothing
    happened at all". It is a valid instrument check and a WORTHLESS power
    control. The honest control is the same triangles over an interval with NO
    resolution in it.
    """
    i, j = sorted(resolved_edge)
    return np.array([i in t and j in t for t in cx.triangles], dtype=bool)


def placebo_interval(cx, omega_t0, omega_t1, resolved_edge):
    """The control that has power: the SAME affected triangles, over an interval
    containing no resolution. Quotes drift for news and money flow; this is how
    much curl that drift alone puts on exactly the triangles the real test reads.

    Returns the affected-triangle curl mean over the placebo interval, which is
    the number a post-resolution reading has to beat.
    """
    c = curl_of_update(cx, omega_t0, omega_t1)
    m = affected_mask(cx, resolved_edge)
    if m.sum() == 0:
        return None
    return dict(placebo_mean=float(c[m].mean()), placebo_sd=float(c[m].std(ddof=1))
                if m.sum() > 1 else float('nan'), n=int(m.sum()))


def score_resolution(cx, omega_before, omega_after, resolved_edge):
    """Affected-triangle curl against same-interval control triangles.

    Returns the two means, the control sd, and the effect in CONTROL SDS -- which
    is the unit the result is stated in. No absolute threshold is used: this
    project has already shipped a bar applied in two incompatible units at a 3.1%
    power ceiling, and the fix is to size against the control's own spread.
    """
    c = curl_of_update(cx, omega_before, omega_after)
    m = affected_mask(cx, resolved_edge)
    if m.sum() == 0 or (~m).sum() < 2:
        return None                       # not enough triangles to compare: refuse
    ctrl_sd = float(c[~m].std(ddof=1))
    aff, ctrl = float(c[m].mean()), float(c[~m].mean())

    # THE DEGENERATE CONTROL, refused rather than divided by. If the control
    # triangles carry no curl AT ALL -- which happens whenever the unaffected part
    # of the update is exactly rating-class -- then ctrl_sd sits at the float64
    # floor and (aff - ctrl)/ctrl_sd explodes. The first version of this function
    # printed +698253480596194.625 control sds on a planted effect and called it a
    # pass. A statistic that returns a spectacular number for a degenerate reason
    # is exactly what STRUCK.md exists to catch, so the scale is refused instead:
    # the effect is reported in control sds only when the control has a spread to
    # measure against, and otherwise as UNSCALED with both means printed.
    scale = max(float(np.abs(np.asarray(omega_after) - np.asarray(omega_before)).max()), 1e-30)
    degenerate = ctrl_sd < 1e-9 * scale
    return dict(affected=aff, control=ctrl, control_sd=ctrl_sd,
                n_affected=int(m.sum()), n_control=int((~m).sum()),
                control_degenerate=degenerate,
                effect_sds=float('nan') if degenerate else (aff - ctrl) / ctrl_sd,
                effect_absolute=aff - ctrl)


def _elo_update(n_teams, winner, loser, K, p):
    dr = np.zeros(n_teams)
    dr[winner] += K * (1.0 - p)
    dr[loser] -= K * (1.0 - p)
    return dr


def demo():
    n = 6
    pairings = [(a, b) for a in range(n) for b in range(a + 1, n)]     # full round robin
    cx = pairing_complex(n, pairings)
    print("(a) THE COMPLEX. %d teams, %d quoted pairings, %d triangles"
          % (n, len(cx.edges), len(cx.triangles)))
    print("    d1 @ d0 = %.3e (must be 0: it is what makes the null exact)"
          % float(np.abs(cx.d1 @ cx.d0).max()))
    assert float(np.abs(cx.d1 @ cx.d0).max()) < 1e-12

    rng = np.random.default_rng(0)
    r = rng.normal(size=n) * 100.0
    omega0 = cx.d0 @ r
    print("(b) THE NULL: a rating-class update has ZERO curl, at any K and any p.")
    worst = 0.0
    for K in (8.0, 16.0, 32.0, 64.0):
        for p in (0.25, 0.5, 0.73):
            om1 = cx.d0 @ (r + _elo_update(n, 0, 1, K, p))
            worst = max(worst, float(curl_of_update(cx, omega0, om1).max()))
    print("    worst |curl| over K x p sweep = %.3e" % worst)
    assert worst < 1e-12, "an Elo update carries curl: the null is not exact"

    print("(c) THE PLANTED NEGATIVE: an update that is NOT any d0 . dr must carry curl,")
    print("    or the statistic cannot tell a potential pricer from anything at all.")
    free = omega0 + rng.normal(size=len(cx.edges)) * 0.5
    print("    rating-class update : max |curl| = %.3e" % worst)
    print("    free edge update    : max |curl| = %.4f" % float(curl_of_update(cx, omega0, free).max()))
    assert float(curl_of_update(cx, omega0, free).max()) > 1e-3
    print("    FIRED: %.1e against %.4f." % (worst, float(curl_of_update(cx, omega0, free).max())))

    print("(d) THE CONTROL. Quotes drift for reasons unrelated to any resolution, so")
    print("    affected triangles are scored against unaffected ones over the SAME")
    print("    interval. Under drift alone the effect must be ~0 control sds.")
    drift = omega0 + rng.normal(size=len(cx.edges)) * 0.05
    s_drift = score_resolution(cx, omega0, drift, (0, 1))
    print("    pure drift, no resolution : effect = %+.3f control sds (n_aff=%d n_ctrl=%d)"
          % (s_drift["effect_sds"], s_drift["n_affected"], s_drift["n_control"]))

    print("    Now a resolution that moves ONLY the affected triangles, off-potential:")
    planted = omega0.copy()
    for ti, t in enumerate(cx.triangles):
        if 0 in t and 1 in t:
            planted[[cx.edges.index(tuple(sorted((t[0], t[1])))),
                     cx.edges.index(tuple(sorted((t[1], t[2]))))]] += 0.4
    s_planted = score_resolution(cx, omega0, planted, (0, 1))
    print("    planted off-potential     : control_degenerate=%s  effect_absolute=%+.4f"
          % (s_planted["control_degenerate"], s_planted["effect_absolute"]))
    print("    (the sd scale is REFUSED here: the control's curl is exactly zero, so")
    print("     dividing by its sd gives a spectacular number for a degenerate reason)")
    assert s_planted["control_degenerate"], "a zero-curl control was not flagged degenerate"
    assert np.isnan(s_planted["effect_sds"]), "a degenerate control still produced an sd scale"
    assert s_planted["effect_absolute"] > 0.01, "the planted off-potential move is not visible"

    print("    Now the realistic case: drift EVERYWHERE plus an off-potential move on")
    print("    the affected triangles, so the control has a genuine spread to scale by.")
    mixed = omega0 + rng.normal(size=len(cx.edges)) * 0.05
    for t in cx.triangles:
        if 0 in t and 1 in t:
            mixed[cx.edges.index(tuple(sorted((t[0], t[1]))))] += 0.4
    s_mixed = score_resolution(cx, omega0, mixed, (0, 1))
    print("    drift + planted           : effect = %+.3f control sds (degenerate=%s)"
          % (s_mixed["effect_sds"], s_mixed["control_degenerate"]))
    assert not s_mixed["control_degenerate"], "a genuinely noisy control was called degenerate"
    assert s_mixed["effect_sds"] > 3.0 > abs(s_drift["effect_sds"]), \
        "the control does not separate a real off-potential move from drift"
    print("    FIRED both ways: %+.3f sds under drift alone, %+.3f sds with an"
          % (s_drift["effect_sds"], s_mixed["effect_sds"]))
    print("    off-potential consequence on top of the same drift.")

    print("(f) THE CONTROL THAT HAS POWER. Under ANY potential model the unaffected")
    print("    triangles have dr = 0 at every vertex, so their update is identically")
    print("    zero -- curl AND signal. The affected-vs-unaffected split therefore")
    print("    tests the noise floor, not an alternative. The control with power is the")
    print("    SAME triangles over an interval containing no resolution.")
    placebo = omega0 + rng.normal(size=len(cx.edges)) * 0.05
    pb = placebo_interval(cx, omega0, placebo, (0, 1))
    real = placebo_interval(cx, omega0, mixed, (0, 1))
    print("    no-resolution interval : affected-triangle curl = %.4f (sd %.4f, n=%d)"
          % (pb["placebo_mean"], pb["placebo_sd"], pb["n"]))
    print("    post-resolution        : affected-triangle curl = %.4f"
          % real["placebo_mean"])
    print("    ratio = %.2fx" % (real["placebo_mean"] / max(pb["placebo_mean"], 1e-12)))
    assert real["placebo_mean"] > 3 * pb["placebo_mean"],         "the placebo interval does not separate a real post-resolution move from drift"
    print("    FIRED: drift alone puts %.4f on these triangles; the planted"
          % pb["placebo_mean"])
    print("    off-potential consequence puts %.4f." % real["placebo_mean"])

    print("(e) REFUSAL. A resolution with too few triangles to compare returns None")
    print("    rather than a number computed from nothing.")
    thin = pairing_complex(3, [(0, 1), (1, 2), (0, 2)])
    print("    3-team complex, %d triangles -> score = %s"
          % (len(thin.triangles), score_resolution(thin, np.zeros(3), np.ones(3), (0, 1))))
    assert score_resolution(thin, np.zeros(3), np.ones(3), (0, 1)) is None
    print("    refused as required.")
    print("ALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    demo()
