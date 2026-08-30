"""C1 -- the VECTOR consequence corpus, and its admission bundle.

WHY THIS EXISTS. Every label in this repository is a scalar point prediction at
position `s-1` (`scale/m3_quintuple.py:311`), which is exactly the shape where
one softmax layer is provably Bayes-optimal (`arXiv:2410.01537`, conceded at
`LOOP_PROMPT.md:34-38`). `STATE.md` items 27-28 already record that the novelty
claim is UNTESTED because no vector-valued label exists here. Eight rounds have
raced a baseline at its proven optimum (MISTAKES.md D-1). This corpus leaves
that regime: the label is `[n, m]`, one consequence value per position.

THE CONSTRUCTION, and why it is not the obvious one. The obvious vector label is
the chain family's own prefix scan `z_p` -- the intermediate state of
`equilibrium_oracle`'s loop (`scale/negation_scope.py:302-303`), which costs
zero new code. `R9_IRENE_PREDICTION.md` §2b shows why that label is worthless,
by direct measurement: `make_equilibrium_batch` sets `a[:, :head+1] = 0.0`, so
for every `p <= head` the scan gives `z_p == b_p` BITWISE, and `b` is written
into the input as `x[:, :, CH_FLIP]`. At `t* = 1` that is 63 of 64 positions
whose label is an identity copy of a channel the arm already holds, readable at
zero hops. It is the exact defect `b[:, s-1] = 0.0` was introduced to remove
(`scale/negation_scope.py:376-387`), re-opened at every position that was never
zeroed. A vector label built that way would print pooled NRMSE two to seven
times better than the scalar ladder ever did, and none of it would be
capability.

C1 fixes the depth instead of the position. The label at `p` is the chain run
over the `t*` positions STRICTLY BEFORE `p`:

    y_p = sum_{h=1..t*}  ( prod_{i=p-h+1..p} a_i ) * b_{p-h}

with `a` Rademacher and `b` Gaussian, the family's own encoding, reusing
CH_DRIVE and CH_FLIP with no new channel. Three consequences, all structural:

  * `h = 0` is absent, so `b_p` never enters `y_p`. This is `b[:, s-1] = 0.0`
    applied at EVERY position -- and applied to the LABEL rather than to the
    input, so the drivers keep full variance and no position's label has `sd 0`.
  * The sum terminates at `t*` exactly. The operator is NILPOTENT, not a
    contraction, so `t*` is a HOP COUNT and not a tolerance -- the same choice
    `make_equilibrium_batch` documents and for the same reason. That is what
    puts the truncation law in closed form with no constant fitted.
  * Every labelled position sums `t*` INDEPENDENT unit-variance terms, so
    `Var(y_p) = t*` at every position alike. No position is privileged, which is
    what `R9_IRENE_PREDICTION.md` §2c's dilution weights need in order not to
    apply -- see `test_every_position_is_equally_deep_and_equally_informative`.

Positions `p < t*` have no full window. Causality forbids fixing that: a
strictly causal operator raised to the `t*` power vanishes on the first `t*`
coordinates, always. So the label band is `p in [t*, s)` and the label is
`[n, s - t*]` -- still vector-valued, and with NO degenerate position rather
than an `[n, s]` carrying `t*` entries of `sd 0`, which is the fourteenth strike
wearing a new shape (MISTAKES.md V-8).

WHAT THIS BUNDLE DOES NOT DECIDE. `R9_IRENE_PREDICTION.md` §2a reads that the
arm writes its pivot term into row `s-1` and no other row
(`scale/m3_quintuple.py:302-306`), so the arms are the same expression at every
other position. That is an ARM defect, it is Venus's own falsifier 5, and it is
Neptune's unit. This corpus removes §2b and §2c and leaves §2a exactly where it
was. Nothing here is evidence about which prediction wins.

PREDICTION 1, RECOMPUTED AGAINST THE LABEL ACTUALLY BUILT. `R9_IRENE_PREDICTION.md`
§6 requires this before that file may be scored: "PREDICTION 1's numbers must be
recomputed from the same formula against the label actually built". Her formula
is `predicted margin = scalar margin * sqrt(w)`, with `w` the share of total
label variance at the one position where the arms differ. C1's per-position
variance is uniform at `t*` (measured below), so `w = 1 / (s - t*)` exactly,
against her `w` of 0.0156 .. 0.0808 on the prefix scan:

    t*    positions   w         sqrt(w)    settled     twin
     1       63       0.015873  0.125988   -0.024827   -0.020288
     2       62       0.016129  0.127000   -0.007715   -0.005528
     8       56       0.017857  0.133631   +0.004809   +0.005381
    32       32       0.031250  0.176777   +0.010545   +0.007711

ALL EIGHT ARE STILL BELOW THE PRE-REGISTERED RESOLUTION 0.027260, and at
`t* = 8` and `t* = 32` they are SMALLER than the numbers she filed. Removing the
free-copy positions does not rescue the contract's prediction; it makes §2a the
sole surviving mechanism. That is the point of building the corpus this way. On
the prefix-scan label a null reading is uninterpretable -- it could be the label
(63 of 64 positions a free copy) or the arm (one row of 64 architecturally
distinct). On C1 the label is uniform by construction, so a null is the arm, and
Venus's own falsifier 5 -- writing `_alpha` at every position rather than only
`s-1` -- becomes a clean one-line test rather than a confound.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scale import negation_scope as NS                        # noqa: E402

S_HOUSE, D_HOUSE = 64, 24       # the E ladder's geometry, test_m3_etasks.py:74
N_DRAW = 2048                   # the shipped n_train / n_eval
RUNGS = (1, 2, 8, 32)           # the e3 rungs, so the lanes are comparable

#: MEASURED sampling bounds at the shipped n, 8 seeds, this machine. Neither is
#: fitted: each is set above the drawn spread and far below the distance to the
#: nearest wrong construction, and both ends are reported by the tests that use
#: them. Provenance discipline is `scale/e_ladder.py:23-27` (MISTAKES.md M-2).
#:
#:   truncation law   worst |NRMSE - closed| over every (t*, k, seed): 0.011145
#:   flipper band     worst |drawn - closed| over every (t*, seed):    0.013556
#:   per-position Var worst relative deviation from t*:                0.045
#:
#: The distance to the wrong construction, for each:
#:   truncation   a label needing no hops reads 0 at k=0 against a closed 1.0
#:   flipper      a flipper-blind label reads exactly 0.0, and the smallest
#:                closed value on the ladder is 0.031746 at t* = 1
#:   variance     the prefix-scan label of R9_IRENE_PREDICTION.md §2b has
#:                Var(z_p) running 1 .. t*, a relative spread of 700% at t* = 8
TOL_TRUNCATION = 0.02
TOL_FLIPPER = 0.02
TOL_VAR_REL = 0.10


def _draw(t_star, seed=0, n=N_DRAW):
    return NS.make_propagate_batch(n, S_HOUSE, D_HOUSE, t_star=t_star,
                                   d_model=16, seed=seed)


# ----------------------------------------------------- obligation 5: shape
@pytest.mark.parametrize("t_star", RUNGS)
def test_every_position_is_equally_deep_and_equally_informative(t_star):
    """THE LABEL IS VECTOR-VALUED AND NO POSITION IS DEGENERATE.

    `sd > 0` per position is the floor -- a constant label makes NRMSE `nan` and
    `float('nan') >= 1.0` is False, so a bare threshold passes it silently
    (MISTAKES.md V-8). This asserts far more: every position's variance is the
    SAME `t*`, which is what makes the pooled reading a reading of the task
    rather than a variance-weighted average dominated by one coordinate."""
    x, y, f, p = _draw(t_star)
    assert y.ndim == 2, f"label is not vector-valued: {tuple(y.shape)}"
    assert y.shape == (N_DRAW, S_HOUSE - t_star), tuple(y.shape)

    per_pos_var = y.var(dim=0, unbiased=False)
    rel = ((per_pos_var - t_star).abs() / t_star)
    print(f"\n  t*={t_star}  label {tuple(y.shape)}  positions={y.shape[1]}")
    print(f"    per-position sd  min={float(y.std(0,unbiased=False).min()):.6f}"
          f"  max={float(y.std(0,unbiased=False).max()):.6f}")
    print(f"    per-position Var vs closed {t_star}: max rel dev={float(rel.max()):.6f}")

    assert float(y.std(dim=0, unbiased=False).min()) > 0.0, "a position is CONSTANT"
    assert torch.isfinite(y).all()
    assert float(rel.max()) < TOL_VAR_REL, float(rel.max())


# ------------------------------------------- obligation 3: k = 0 is the mean
@pytest.mark.parametrize("t_star", RUNGS)
def test_the_zero_hop_reading_is_the_mean_predictor_identically(t_star):
    """NO SHARE OF THE LABEL IS LEGIBLE WITHOUT HOPS, and it is structural.

    `R_0` is the empty sum, so it is bitwise zero, and `nrmse(0, y)` is
    `sqrt(1 + mean(y)**2 / var(y))` -- an identity that is `>= 1.0` for every
    draw, not a number that happens to land there. This is the clause
    `make_equilibrium_batch` bought with `b[:, s-1] = 0.0` after the RED gate
    aborted three of five rungs with INSTRUMENT BROKEN, and it is what
    `R9_IRENE_PREDICTION.md` PREDICTION 3 says a vector label cannot have."""
    x, y, f, p = _draw(t_star)
    r0 = NS.propagate_hop_reading(x, 0, t_star=t_star)
    assert torch.equal(r0, torch.zeros_like(r0)), "R_0 is not the empty sum"
    got = NS.nrmse(r0, y)
    print(f"\n  t*={t_star}  NRMSE(R_0, y) = {got:.9f}")
    assert got >= 1.0, got
    assert abs(got - 1.0) < TOL_TRUNCATION, got


# ------------------------------ obligations 1, 2, 4: one function, closed form
@pytest.mark.parametrize("t_star", RUNGS)
def test_the_truncation_ladder_follows_its_closed_form(t_star):
    """ONE FUNCTION SERVES THE LADDER AND THE LABEL, and the curve is closed.

    `scale/e4_harmonic.py:191` -- "One function for both, because they are the
    same object." `propagate_oracle` IS `propagate_hop_reading` at `k = t*`, so
    the ladder cannot drift from the label it is read against.

    NILPOTENT REGIME, DELIBERATELY. The sum terminates at `t*`, so the discarded
    tail is `t* - k` independent unit-variance terms against a label that is
    `t*` of them, and the truncation NRMSE is `sqrt((t*-k)/t*)` EXACTLY, with no
    constant fitted. A contraction label would need the relative form
    `got <= zeroth * L**k` instead (`test_m3_etasks.py:208`) because NRMSE
    normalises by `std(y)` and not by initial error -- that form is not used
    here because it is not the regime this corpus is in."""
    x, y, f, p = _draw(t_star)
    assert torch.equal(NS.propagate_oracle(x, f, p),
                       NS.propagate_hop_reading(x, t_star, t_star=t_star)), \
        "the oracle and the ladder are not the same function"

    prev, worst = None, 0.0
    print(f"\n  t*={t_star}")
    for k in NS.e_ladder_ks(t_star):
        if k > t_star:
            continue
        got = NS.nrmse(NS.propagate_hop_reading(x, k, t_star=t_star), y)
        closed = math.sqrt((t_star - k) / t_star)
        worst = max(worst, abs(got - closed))
        print(f"    k={k:>3}  NRMSE={got:.6f}  closed={closed:.6f}  dev={abs(got-closed):.2e}")
        assert abs(got - closed) < TOL_TRUNCATION, (k, got, closed)
        if prev is not None:
            assert got < prev, (k, got, prev)      # monotone tightening
        prev = got

    # k = 1 bounded away from the label; exact at the full budget, bitwise.
    if t_star > 1:
        assert NS.nrmse(NS.propagate_hop_reading(x, 1, t_star=t_star), y) > 0.5
    assert torch.equal(NS.propagate_hop_reading(x, t_star, t_star=t_star), y)
    #: NILPOTENCY, SHOWN RATHER THAN STATED. `e_ladder_ks` prints one rung PAST
    #: the dial precisely so termination is visible: a contraction would keep
    #: tightening there, a nilpotent operator is already exact and stays exact.
    assert t_star + 1 in NS.e_ladder_ks(t_star)
    assert torch.equal(NS.propagate_hop_reading(x, t_star + 1, t_star=t_star), y)
    print(f"    worst |NRMSE - closed| = {worst:.6f}   k=t*+1 still bitwise exact")


# --------------------------------------- obligation 6: the do()-bit moves it
@pytest.mark.parametrize("t_star", RUNGS)
def test_the_label_moves_with_its_do_bit_at_exactly_the_positions_it_should(t_star):
    """THE MOVEMENT TEST, AND IT IS THE CAUSALITY CHECK TOO.

    A ratio band is the WRONG-TASK check, not the anti-vacuity check
    (FINDINGS F). This asserts the exact support of the perturbation: negating
    the driver at `f` must move the label at exactly the `t*` positions
    `f+1 .. f+t*` and leave every other position BITWISE unchanged. A label that
    ignored its drivers passes no part of this; a label that leaked a future
    position fails the bitwise half."""
    x, y, f, p = _draw(t_star)
    xf = x.clone()
    xf[:, f, NS.CH_FLIP] = -xf[:, f, NS.CH_FLIP]
    delta = NS.propagate_oracle(xf, f, p) - y

    band0 = t_star                                   # label column 0 is x-position t*
    moved_cols = torch.nonzero((delta != 0).any(dim=0)).flatten().tolist()
    expect = [q - band0 for q in range(f + 1, f + t_star + 1)]
    print(f"\n  t*={t_star}  f={f}  moved label columns={len(moved_cols)} expected={len(expect)}")
    assert moved_cols == expect, (moved_cols[:8], expect[:8])

    got = float(delta.abs().mean()) / float(y.abs().mean())
    closed = NS.propagate_flipper_dependence(S_HOUSE, t_star=t_star)
    print(f"    flipper_dependence drawn={got:.6f}  closed={closed:.6f}"
          f"  dev={abs(got-closed):.2e}")
    assert abs(got - closed) < TOL_FLIPPER, (got, closed)
    assert closed > 0.0, "a task declaring 0.0 must ship a do()-bit movement test"


def test_the_shipped_flipper_band_is_too_loose_for_this_family():
    """A RECORDED WEAKNESS, measured, not papered over.

    `bar_verdict`'s default `flipper_tol = 0.05` was calibrated for `y = payload
    * sign`, whose exact ratio is 2.0. A vector label spreads one driver's
    influence over `s - t*` positions, so the exact ratio here is
    `2*sqrt(t*)/(s-t*)`, which at `t* = 1` is 0.031746 -- SMALLER than the
    default tolerance. The band therefore accepts 0.0, a flipper-blind label, at
    the two shallow rungs. That is a gate satisfied by construction
    (MISTAKES.md V-10), so it is asserted here rather than left to be
    discovered: the anti-vacuity duty is carried by the movement test above, and
    the tolerance this family needs is TOL_FLIPPER = 0.02, which accepts the
    drawn spread (worst 0.013556) and rejects 0.0 at every rung."""
    loose = []
    for t_star in RUNGS:
        exact = NS.propagate_flipper_dependence(S_HOUSE, t_star=t_star)
        blind_accepted = abs(0.0 - exact) <= 0.05
        print(f"  t*={t_star:>2} exact={exact:.6f}  shipped 0.05 band accepts a "
              f"flipper-blind label: {blind_accepted}")
        if blind_accepted:
            loose.append(t_star)
        assert abs(0.0 - exact) > TOL_FLIPPER, (t_star, exact)
    assert loose == [1, 2], loose


# ------------------------------------------------------- the registration
def test_the_family_is_registered_and_routes_to_its_own_ladder():
    """REGISTRATION WITHOUT ADMISSION IS A DEFECT IN ITSELF (MISTAKES.md D-4).

    `E_T_STAR` has no entry for `impact`, `impact_hetero` or `e4prime`, and two
    consumers of it fail DIFFERENTLY on the missing key -- `e_ladder.py:143`
    raises, `m3_capability.py:269` degrades silently. And `e_hop_reading` falls
    through to the CHAIN family's reading for any task it does not name, so an
    unrouted task prints another family's ladder against its own label. Both
    holes are closed here at registration time rather than left for a caller."""
    for t_star in RUNGS:
        name = f"c1_propagate_t{t_star}"
        assert name in NS.M3_TASKS, name
        assert len(NS.M3_TASKS[name]) == 4, name          # test_m3_etasks.py:97
        assert name in NS.E_T_STAR, f"{name} has no difficulty dial"
        assert NS.E_T_STAR[name](S_HOUSE) == t_star
        for forbidden in ("_sd", "_task", "_k", "_b"):    # m3_quintuple.py:561-577
            assert forbidden not in name, (name, forbidden)

        x, y, f, p = _draw(t_star, n=256)
        routed = NS.e_hop_reading(name, x, f, p, 0)
        assert routed.shape == y.shape, (
            f"{name} routes to another family's ladder: {tuple(routed.shape)} "
            f"against a label of {tuple(y.shape)}")


def test_the_registration_calibrates_through_its_own_entry():
    """A REGISTRATION THAT CANNOT PASS `calibrate_bar` THROUGH ITS OWN ENTRY IS
    NOT A REGISTRATION. Every hook is taken from `M3_TASKS`, so a mis-paired
    builder, oracle or feature function fails here rather than in a run."""
    t_star = 8
    name = f"c1_propagate_t{t_star}"
    batch_fn, oracle_fn, feature_fn, fd_fn = NS.M3_TASKS[name]
    cal = NS.calibrate_bar(n=512, s=S_HOUSE, d=D_HOUSE, steps=150,
                           batch_fn=batch_fn, oracle_fn=oracle_fn,
                           feature_fn=feature_fn)
    for k, v in cal.items():
        print(f"    {k:>20} {v:.6f}")
    ok, why = NS.bar_verdict(cal, flipper_dependence=fd_fn(S_HOUSE),
                             flipper_tol=TOL_FLIPPER)
    assert ok, (why, cal)


@pytest.mark.parametrize("t_star", RUNGS)
def test_the_bundle_rejects_the_prefix_scan_label_it_was_written_against(t_star):
    """THE PLANTED NEGATIVE. A control that cannot reject the wrong
    construction is not a control.

    The wrong construction is not hypothetical: it is the cheapest vector label
    available, it is what "drop the `[:, s-1]` index" most naturally means, and
    `R9_IRENE_PREDICTION.md` §2b is a filed prediction that building it would
    print numbers two to seven times better than the scalar ladder while
    measuring nothing. This test builds that exact label -- the prefix scan of
    `equilibrium_oracle`'s own loop over `make_equilibrium_batch`'s tensor --
    and runs C1's admission clauses against it.

    IT ALSO REPRODUCES §2b INDEPENDENTLY. The bitwise copy counts asserted here
    were derived by Venus from `a[:, :head+1] = 0.0`; they are re-measured here
    from a draw, and they agree: 63, 62, 56 and 32 of 64.

    TWO CLAUSES THAT FAIL DIFFERENTLY. The zero-hop clause rejects at all four
    rungs. The equal-variance clause rejects at three: at `t* = 1` the prefix
    scan's per-position variance IS nearly uniform, because 63 of its 64
    positions are the same unit-variance copy -- uniformly uninformative rather
    than uniformly deep. A bundle resting on the variance clause alone would
    have admitted the worst rung of the worst construction."""
    x, _y, f, p = NS.make_equilibrium_batch(N_DRAW, S_HOUSE, D_HOUSE,
                                            t_star=t_star, d_model=16, seed=0)
    a, b = x[:, :, NS.CH_DRIVE], x[:, :, NS.CH_FLIP]
    z, cols = torch.zeros(N_DRAW), []
    for i in range(S_HOUSE):
        z = a[:, i] * z + b[:, i]
        cols.append(z.clone())
    scan = torch.stack(cols, dim=1)                      # the prefix-scan label

    head = S_HOUSE - 1 - t_star
    copies = int((scan[:, :head + 1] == b[:, :head + 1]).all(dim=0).sum())
    var_rel = float(((scan.var(dim=0, unbiased=False) - t_star).abs() / t_star).max())
    zero_hop = NS.nrmse(b, scan)          # its zero-hop reading is its own driver

    print(f"\n  t*={t_star}  prefix scan: z_p == b_p BITWISE at {copies}/{S_HOUSE}"
          f" positions (R9_IRENE_PREDICTION.md §2b)")
    print(f"    C1 equal-variance clause (< {TOL_VAR_REL}): {var_rel:.4f}"
          f"  -> {'admits' if var_rel < TOL_VAR_REL else 'REJECTS'}")
    print(f"    C1 zero-hop clause (>= 1.0):            {zero_hop:.6f}"
          f"  -> {'admits' if zero_hop >= 1.0 else 'REJECTS'}")

    # `a` is zeroed at and before `head = s-1-t*`, so `z_p == b_p` for the
    # `head + 1 = s - t*` positions `0 .. head`. R9_IRENE_PREDICTION.md §2b
    # reports 63 of 64 at t*=1 and 56 of 64 at t*=8; both are `s - t*`.
    assert copies == S_HOUSE - t_star, (copies, S_HOUSE - t_star)
    assert zero_hop < 1.0, (
        "the zero-hop clause ADMITS the prefix-scan label -- C1's admission "
        "bundle does not discriminate against the construction it was written "
        "to rule out")
    # And C1's own label, on the same clause, at the same rung.
    _cx, cy, _cf, _cp = _draw(t_star)
    assert NS.nrmse(NS.propagate_hop_reading(_cx, 0, t_star=t_star), cy) >= 1.0
