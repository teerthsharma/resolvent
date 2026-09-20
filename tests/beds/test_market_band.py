"""tests/beds/test_market_band.py

Lands Defect 4 on ceqjepa/beds/market_band.py (the market-band existence
bed, named MARKET-BAND in the module's own docstring -- see NAMING below):
before this file, tests/beds/ held test_bed_1.py, test_bed_h.py and
test_bed_k.py and nothing named market_band -- the module's only assertions
lived in its __main__ block, which pytest never runs. NO RED, NO FINDING: the
band claim ("the no-trade band derives from the cost arithmetic, not a
stated rule") was the one reference the bed reproduces non-trivially, and it
was unguarded.

INTERFACE, AS FOUND (another lane edited this bed in the same round; this
file writes to the module as it now reads, not as the defect report
described it). MarketBandBed(q, c, dtype).multiplier(f, yes_happens,
cost_on="notional") takes a SIGNED f (f>=0 long-YES at price q, f<0 long-NO
at price 1-q, `yes_happens` the realized outcome) -- `won` from the earlier
revision is gone, replaced by `yes_happens` plus f's own sign. .refuses(p,
eps=1e-6) is now DERIVED from multiplier()'s own marginal sign at f=0+/-eps,
not a stated |p-q|<=c rule. Because the fixed multiplier is two-sided
fair-odds (price = q or 1-q depending on side, drag = c*|f|/price, so the
cost is never sign-inconsistent the way an unsigned c*f/q term is), its
marginal derivative works out in closed form to sign(p-q-c) on the buy side
and sign(q-p-c) on the sell side -- so the TRUE band is exactly [q-c, q+c]
for every q, not only q=0.5. That happens to be the same interval refuses()
stated before the fix; what changed is that it is now PRODUCED by the
multiplier's arithmetic instead of asserted independently of it, and the
previous mismatch was between that stated rule and the *old* (even-money-
only) multiplier, not between it and any external reference. Every test
below re-derives [q-c, q+c] itself (from a first-principles marginal
argument, restated in _derived_band_edges) rather than importing it from the
module's docstring, and cross-checks it against a finite difference taken
through multiplier() directly (test 3), so an algebra slip in either place
cannot manufacture a false green.

NAMING (Defect 5). "B3" collided with two other things in this tree before
the rename found in the module docstring: CONTRACT.md:166 ("B3 Moebius
inversion on incidence algebras") and the census row in
docs/canon/CHARTER.md:229 / docs/canon/04_BEDS_AND_INSTRUMENTS.md ("B3:
tests coupled to prose"). Recorded here for whoever last touched the naming,
not re-litigated.

WHAT IS GUARDED:

  1. refuses(p) matches the true band [q-c, q+c] at THREE q (0.50, 0.40,
     0.60), not only q=0.50 where a stated rule and a derived one are most
     likely to look the same by accident.
  2. f_star's zero/nonzero boundary sits exactly at those edges, same three q.
  3. The band is grounded directly in multiplier()'s own marginal sign (a
     finite difference on the public method itself, both the buy side and
     the sell side, per Rule 2's "counted by instrumenting the branch"), not
     merely asserted from a closed form.
  4. The refuser beats a forced cost-blind bettor's expected per-bet growth
     strictly inside the band, at the same three q.
  5. The exact (q, c, p) = (0.40, 0.03, 0.47) the defect report named: under
     the OLD even-money-only multiplier this was a loss-making forced trade
     (g = -1.128834e-02) hiding behind a correctly-shaped but merely-stated
     refuses() rule. Under the fixed two-sided multiplier, p=0.47 is
     genuinely outside [0.37, 0.43.] (unlike the report's own worked
     example, which located it inside the *old* buggy multiplier's implied
     band of [0.4625, 0.5375]) and f_star's stake there is now profitable,
     not loss-making -- the fix changed which of those two things was true,
     and this test is the one that would go red if it regressed back to
     losing money on this input.

RULE 2 (guarded-entry counts, by instrumentation, not estimate). Every loop
below counts each (q, p) or (q, edge, direction) triple it actually
evaluates and asserts the count against what the grid construction
promises. Counts: band-membership scan = 3 q x 41 p = 123; f_star boundary
probes = 3 q x 4 probes = 12; multiplier-derivative grounding = 3 q x 2
edges x 3 probes = 18; refuser-vs-forced-blind = 3 q x 1 point = 3; the
named counterexample = 1.

Run: python -m pytest tests/beds/test_market_band.py -v
"""
from __future__ import annotations

import math

import pytest

from ceqjepa.beds.market_band import MarketBandBed

C = 0.03
QS = (0.50, 0.40, 0.60)  # 0.50 is the case a stated-vs-derived mismatch can
                          # hide behind; 0.40 and 0.60 are where it cannot.


def _derived_band_edges(q, c=C):
    """The true no-trade band, re-derived here independently of the module:
    for the two-sided fair-odds multiplier (price=q long-YES, price=1-q
    long-NO, drag=c*|f|/price), the marginal log-growth derivative at f=0
    works out to (p-q-c)/price on the buy side and (q-p-c)/price on the
    sell side (both proportional to a positive price, so only the numerator
    sign matters). Neither is profitable, i.e. refuse, exactly on
    [q-c, q+c]."""
    return q - c, q + c


def _g(bed, p, f):
    """Expected per-bet log-growth at signed stake f, via bed.multiplier()
    itself -- the actual guarded code, per Rule 2."""
    win = bed.multiplier(f, True)
    lose = bed.multiplier(f, False)
    if win <= 0.0 or lose <= 0.0:
        return float("-inf")
    return p * math.log(win) + (1.0 - p) * math.log(lose)


def test_no_trade_band_matches_q_minus_c_q_plus_c_at_more_than_one_q():
    """refuses(p) must classify every p on a wide grid the same way the true
    band [q-c, q+c] does, at three q -- a check at q=0.5 alone cannot tell a
    derived rule from a merely-stated one that happens to look the same
    there."""
    p_grid = [round(0.30 + 0.01 * i, 2) for i in range(41)]  # 0.30..0.70
    evaluated = 0
    mismatches = []
    for q in QS:
        bed = MarketBandBed(q=q, c=C)
        lo, hi = _derived_band_edges(q)
        for p in p_grid:
            derived_refuse = lo <= p <= hi
            stated_refuse = bed.refuses(p)
            evaluated += 1
            if derived_refuse != stated_refuse:
                mismatches.append((q, p, derived_refuse, stated_refuse))

    assert evaluated == 123, f"grid shrank: evaluated {evaluated} (q, p) pairs, expected 123"
    assert not mismatches, (
        f"refuses() disagrees with the true band [q-c, q+c] on "
        f"{len(mismatches)}/{evaluated} guarded entries (q, p, true_refuse, "
        f"stated_refuse) = {mismatches[:8]}{'...' if len(mismatches) > 8 else ''}"
    )
    for q in QS:
        bed = MarketBandBed(q=q, c=C)
        inside = sum(1 for p in p_grid if bed.refuses(p))
        outside = sum(1 for p in p_grid if not bed.refuses(p))
        assert inside > 0 and outside > 0, (
            f"q={q}: grid never exercised both the refuse and the trade "
            f"branch (inside={inside}, outside={outside}) -- vacuous test"
        )


def test_band_edges_match_f_star_zero_boundary_at_more_than_one_q():
    """f_star's zero/nonzero boundary must sit exactly at q-c and q+c, at
    three q."""
    evaluated = 0
    for q in QS:
        bed = MarketBandBed(q=q, c=C)
        lo, hi = _derived_band_edges(q)
        for p, want_zero in ((lo, True), (hi, True), (lo - 0.01, False), (hi + 0.01, False)):
            f = bed.f_star(p)
            evaluated += 1
            if want_zero:
                assert f == 0.0, f"q={q}: f_star({p}) = {f}, expected exactly 0.0 (inside true band)"
            else:
                assert f != 0.0, f"q={q}: f_star({p}) = {f}, expected nonzero (outside true band)"
    assert evaluated == 12, f"boundary-probe count drifted: {evaluated}, expected 12"


def test_band_edges_are_where_the_multiplier_marginal_derivative_crosses_zero():
    """Grounds both edges directly in bed.multiplier(), not in a closed form
    taken on faith: a finite difference of g(f) through the public
    multiplier() must be negative strictly inside each edge, ~0 at it, and
    positive strictly outside -- checked on the buy side at q+c and the sell
    side at q-c, at three q."""
    h = 1e-6
    tol = 1e-3  # h-scale finite-difference noise is far below this
    evaluated = 0
    for q in QS:
        bed = MarketBandBed(q=q, c=C)
        lo, hi = _derived_band_edges(q)

        # buy side: forward difference (g(h) - g(0)) / h at hi
        for p, expect in ((hi - 0.02, "not_profitable"), (hi, "zero"), (hi + 0.02, "profitable")):
            deriv = (_g(bed, p, h) - _g(bed, p, 0.0)) / h
            evaluated += 1
            if expect == "profitable":
                assert deriv > tol, f"q={q}, p={p} (buy side): deriv={deriv}, expected > 0"
            elif expect == "not_profitable":
                assert deriv < -tol, f"q={q}, p={p} (buy side): deriv={deriv}, expected < 0"
            else:
                assert abs(deriv) <= tol, f"q={q}, p={p} (buy side): deriv={deriv}, expected ~0"

        # sell side: backward difference (g(0) - g(-h)) / h at lo -- negative
        # means shorting increases g, i.e. shorting is profitable.
        for p, expect in ((lo + 0.02, "not_profitable"), (lo, "zero"), (lo - 0.02, "profitable")):
            deriv = (_g(bed, p, 0.0) - _g(bed, p, -h)) / h
            evaluated += 1
            if expect == "profitable":
                assert deriv < -tol, f"q={q}, p={p} (sell side): deriv={deriv}, expected < 0"
            elif expect == "not_profitable":
                assert deriv > tol, f"q={q}, p={p} (sell side): deriv={deriv}, expected > 0"
            else:
                assert abs(deriv) <= tol, f"q={q}, p={p} (sell side): deriv={deriv}, expected ~0"
    assert evaluated == 18, f"evaluated {evaluated} (q, edge, probe) triples, expected 18"


def test_refuser_beats_forced_cost_blind_bettor_inside_the_band():
    """The part that is right and must not be disturbed: refusing (a fixed
    per-bet growth of 0.0) strictly beats the forced cost-blind bettor's own
    expected per-bet log-growth, for a point strictly inside the true band,
    at three q. multiplier()'s drag is c*|f|/price (always non-negative), so
    unlike the pre-fix module this needs no restriction to p > q."""
    evaluated = 0
    for q in QS:
        bed = MarketBandBed(q=q, c=C)
        lo, hi = _derived_band_edges(q)
        p = lo + 0.25 * (hi - lo)  # strictly inside, not the trivial center
        assert p != q, "test setup error: interior point degenerated to q (f_blind would be 0)"
        forced_growth = bed.forced_blind_growth(p)
        refuser_growth = 0.0  # bed.refuser_wealth is a fixed point: 0 log-wealth for any n
        evaluated += 1
        assert forced_growth < refuser_growth, (
            f"q={q}, p={p}: forced cost-blind growth {forced_growth} does not "
            f"lose to the refuser's {refuser_growth}"
        )
    assert evaluated == 3, f"evaluated {evaluated} q-values, expected 3"


def test_constructed_case_q0_40_p0_47_from_the_defect_report_now_reads_profitable():
    """The exact (q, c, p) the defect report named as a loss under the OLD
    even-money-only multiplier (g = -1.128834e-02 at f_star = +0.0667,
    strictly worse than refusing). p=0.47 is outside the true band
    [0.37, 0.43] at q=0.40 (it is not inside it, unlike the report's own
    reading of the pre-fix multiplier's implied band [0.4625, 0.5375]), so
    f_star trades here, and under the fixed two-sided multiplier that trade
    must be profitable, not loss-making -- the one number this test would
    catch regressing back to negative."""
    q, c, p = 0.40, 0.03, 0.47
    lo, hi = _derived_band_edges(q, c)
    assert not (lo <= p <= hi), f"test setup error: p={p} is inside [{lo}, {hi}], expected outside"

    bed = MarketBandBed(q=q, c=c)
    assert bed.refuses(p) is False, f"refuses({p}) at q={q}, c={c} must be False: p is outside [{lo}, {hi}]"

    f = bed.f_star(p)
    assert f != 0.0, f"f_star({p}) at q={q}, c={c} must trade (p is outside the band)"
    g = _g(bed, p, f)
    assert g > 0.0, (
        f"f_star({p})={f} at q={q}, c={c} gives g={g}; expected a profitable "
        f"stake (the pre-fix multiplier gave g=-1.128834e-02 on this exact input)"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
