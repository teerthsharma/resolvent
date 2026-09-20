"""ceqjepa/sharpness.py -- the cross-entropy decomposition, as a live instrument.

WHAT THIS IS FOR. A read q: X -> simplex_K is scored by cross-entropy, and a
cross-entropy is a SUM of four things that move independently. Reporting only
the sum is how a model spends 4 h 43 m of GPU time paying 1.397 nats of
sharpness to collect 0.120 nats of information [MEASUREMENT -- a real run;
this file carries no run id for it, see ASSAY C1] and nothing in the loop notices.
The identity below splits the sum, and every term is computable at every eval
from the same (q, y) the loss already has in hand.

THE IDENTITY. For (X,Y) ~ D with Y in {1..K}, pi_b = P(Y=b), H(pi) the marginal
entropy, and every coordinate of q bounded below by epsilon > 0:

    KL   = KL(pi || E q(X))                                   marginal mismatch
    J(q) = sum_b pi_b [ ln E q_b(X) - E ln q_b(X) ]           SHARPNESS >= 0 (Jensen),
                                                              = 0 iff each q_b is a.s. constant
    I_q  = sum_b pi_b [ E(-ln q_b(X)) - E(-ln q_b(X)|Y=b) ]   LABEL COVARIANCE of the read

    (i)   E[-ln q_Y(X)] = H(pi) + KL + J(q) - I_q             EXACT, not a bound
    (iii) beats the MARGINAL predictor iff  I_q > J(q) + KL
          beats the UNIFORM  predictor iff  J(q) + KL - I_q < ln K - H(pi)

(i) is an algebraic identity of the empirical means, so on any finite (q, y) it
holds to roundoff, not to sampling error: `residual` below is asserted < 1e-8
[THRESHOLD -- a tolerance, not itself construction-property or measurement]
and runs ~1e-16 in float64 [CONSTRUCTION-PROPERTY -- true for any (q,y),
by algebra, never a diagnostic of the model]. Verified against a real run at
1e-4 in the reported precision:
2.3406 = 1.0609 + 0.0025 + 1.3967 - 0.1196 (held-out PPL 10.39, marginal
control 2.89, chance 4.00 -- i.e. that read LOST to a predictor that never
looked at the input, while its cross-entropy alone looked merely mediocre).
[MEASUREMENT -- same real run as the opening paragraph (1.397≈1.3967 nats,
0.120≈0.1196 nats); no run id in this file either time, see ASSAY C1]

THE TRAP THIS FILE EXISTS TO CATCH, and case (d) of the self-check demonstrates
rather than asserts it: sharpening a perfectly calibrated read with a
temperature below 1 is a strictly monotone per-row map, so ARGMAX ACCURACY IS
UNCHANGED -- every ranking metric, top-1, top-k, AUC on the argmax, is frozen --
while J(q) grows without bound and I_q does not. The read crosses from
beats_marginal True to False with its accuracy identical to the digit. A model
can rank perfectly and still be worse than a constant.

DECIDING AN INEQUALITY NEEDS A STANDARD ERROR. I_q > J + KL is a decision, and
`bootstrap_se` resamples ITEMS and PAIRS the resample -- J, I_q and the margin
I_q - (J + KL) are recomputed on the SAME resampled indices, because they are
functions of the same items and an unpaired bootstrap inflates the margin's SE.
Report margin +/- se_margin, never the sign alone.

THE PRE-REGISTERED BAR (MUST-FIRE, see PREREG BAR below). `d["beats_marginal"]`
is sign(margin), and sign alone is not a verdict -- a margin of +0.01 at
se_margin=0.02 is 0.5 SE from zero and is noise, not a win. `passes_bar(d, se)`
is the fixed rule margin > BAR_SIGMA(2) * se_margin, `kill_fired(d, se)` is
margin <= KILL_SIGMA(1) * se_margin. Read those, never `beats_marginal`, when
the question is whether a read has actually cleared the bar.

USAGE at eval time:
    d = decompose(q, y, K)                 # q [n,K] probabilities, y [n] int
    se = bootstrap_se(q, y, K)
    print(format_line(d, se), "PASS" if passes_bar(d, se) else "KILL" if kill_fired(d, se) else "GRAY")
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "decompose",
    "format_line",
    "bootstrap_se",
    "margin_sigma",
    "passes_bar",
    "kill_fired",
    "BAR_SIGMA",
    "KILL_SIGMA",
    "ConstructionPropertyAsBarError",
]


def _prep(q, y, K, eps):
    """Validate, cast to float64, floor at eps. Returns (q, logq, y, K)."""
    q = torch.as_tensor(q, dtype=torch.float64)
    y = torch.as_tensor(y).to(torch.int64).reshape(-1)
    assert q.dim() == 2, f"q must be [n,K], got {tuple(q.shape)}"
    assert q.shape[0] == y.shape[0] > 0, f"q has {q.shape[0]} rows, y has {y.shape[0]}"
    K = q.shape[1] if K is None else int(K)
    assert K == q.shape[1], f"K={K} but q has {q.shape[1]} columns"
    assert torch.isfinite(q).all(), "q has non-finite entries"
    assert (q >= 0).all(), "q has negative entries"
    assert 0 <= int(y.min()) and int(y.max()) < K, f"y outside [0,{K})"
    err = (q.sum(1) - 1).abs().max()  # CONSTRUCTION-PROPERTY: how close the input rows are to the simplex
    assert err < 1e-4, f"rows of q are not on the simplex (max |sum-1| = {err:.3e})"  # THRESHOLD, unclassifiable under C1
    # the epsilon of the theorem: a floor for the log. No renormalisation -- the
    # identity holds for any positive q, and renormalising would move the read.
    q = q.clamp_min(eps)  # eps: THRESHOLD (instrument floor), unclassifiable under C1
    return q, q.log(), y, K


def _terms(q, logq, y, K):
    """(ce, H_pi, kl, sharpness, i_q) as float64 scalars. Classes absent from y
    carry pi_b = 0 and an undefined conditional, so they are dropped: 0 * NaN."""
    n = y.numel()
    counts = torch.bincount(y, minlength=K).to(torch.float64)
    pi = counts / n
    p = counts > 0
    v = logq.gather(1, y[:, None]).squeeze(1)  # ln q_{y_i}(x_i), per item
    ce = -v.mean()
    H = -(pi[p] * pi[p].log()).sum()
    qbar = q.mean(0)  # E q_b(X)
    kl = (pi[p] * (pi[p].log() - qbar[p].log())).sum()
    e_logq = logq.mean(0)  # E ln q_b(X)
    sharp = (pi[p] * (qbar[p].log() - e_logq[p])).sum()
    cond = torch.zeros(K, dtype=torch.float64, device=v.device).index_add_(0, y, v) / counts.clamp_min(1)
    i_q = (pi[p] * (cond[p] - e_logq[p])).sum()  # E(-ln q_b) - E(-ln q_b | Y=b)
    return ce, H, kl, sharp, i_q


def decompose(q, y, K=None, eps=1e-12, tol=1e-8, construction_property=False):
    # eps, tol: THRESHOLDs (instrument floor / residual tolerance) -- neither
    # CONSTRUCTION-PROPERTY nor MEASUREMENT, see ASSAY C1's unclassifiable set.
    # construction_property: caller-supplied tag, propagated into the returned
    # dict so passes_bar/kill_fired can refuse to treat this read as a bar.
    """Split E[-ln q_Y(X)] into H(pi) + KL + J(q) - I_q on one eval batch.

    q [n,K] probabilities, y [n] labels in [0,K). Every returned field a float
    (three are bools). `residual` is the identity gap and is asserted tiny.
    """
    q, logq, y, K = _prep(q, y, K, eps)
    ce, H, kl, sharp, i_q = _terms(q, logq, y, K)
    residual = float(ce - (H + kl + sharp - i_q))  # CONSTRUCTION-PROPERTY: algebraic identity gap, true for any (q,y)
    assert abs(residual) < tol, f"identity violated by {residual:.3e} -- bug in this file, not in the model"
    ce, H, kl, sharp, i_q = (float(t) for t in (ce, H, kl, sharp, i_q))
    margin = i_q - sharp - kl  # > 0 iff the read beats the marginal predictor
    return {
        "ce": ce,
        "H_pi": H,
        "kl": kl,
        "sharpness": sharp,
        "i_q": i_q,
        "margin": margin,
        "residual": residual,
        "beats_marginal": bool(i_q > sharp + kl),
        "beats_uniform": bool(sharp + kl - i_q < math.log(K) - H),
        "ppl": math.exp(ce),
        "ppl_marginal": math.exp(H),
        "ppl_chance": float(K),
        "construction_property": bool(construction_property),  # ASSAY C1 tag: see passes_bar/kill_fired guard below
    }


def bootstrap_se(q, y, K=None, n_boot=200, seed=0, eps=1e-12):
    """Paired item bootstrap SEs for sharpness, i_q and the margin i_q-(J+KL).

    The three are recomputed on the SAME resampled indices every draw; the
    margin's SE is the one the I_q > J + KL decision actually needs.
    """
    q, logq, y, K = _prep(q, y, K, eps)
    n = y.numel()
    g = torch.Generator().manual_seed(int(seed))
    draws = torch.empty(n_boot, 3, dtype=torch.float64)
    for b in range(n_boot):
        idx = torch.randint(n, (n,), generator=g)
        _, _, kl_, sharp_, i_ = _terms(q[idx], logq[idx], y[idx], K)
        draws[b] = torch.stack([sharp_, i_, i_ - sharp_ - kl_])
    se = draws.std(0, unbiased=True)
    return {
        "se_sharpness": float(se[0]),
        "se_i_q": float(se[1]),
        "se_margin": float(se[2]),
        "n_boot": float(n_boot),
    }


# --------------------------------------------------------------------------- #
# PREREG BAR -- this file had none (a margin's sign alone is not a decision,
# only its point estimate is). Set BEFORE looking at any estimated-operator
# result, so the bed can be graded, not eyeballed.
#
#   BAR  (MUST-FIRE to count as a pass): margin >  BAR_SIGMA  * se_margin.
#   KILL (fires the read is dead):       margin <= KILL_SIGMA * se_margin.
#
# Between the two is a gray zone: neither a beat nor a confident loss. Both
# thresholds are read off the SAME paired bootstrap `bootstrap_se` already
# returns, because a margin's sign without its SE is exactly the "unsolicited
# vibe" this file's identity split exists to replace.

BAR_SIGMA = 2.0  # THRESHOLD (pre-registered design decision) -- unclassifiable under ASSAY C1's two-way split
KILL_SIGMA = 1.0  # THRESHOLD, same note


class ConstructionPropertyAsBarError(ValueError):
    """Raised when a figure tagged construction_property=True (an oracle demo,
    a hand-picked example, a self-test on synthetic data -- never a model read)
    is handed to passes_bar/kill_fired as though it were a bar decision.
    LAWS: 'A construction property is never a bar.'"""


def margin_sigma(d, se):
    """margin expressed in bootstrap standard errors: margin / se_margin."""
    return d["margin"] / se["se_margin"]


def _guard_not_construction_property(d, caller):
    if d.get("construction_property"):
        raise ConstructionPropertyAsBarError(
            f"{caller}: this figure is tagged construction_property=True -- "
            "it is a property of how the example was built (an oracle read, a "
            "synthetic draw, a hand-picked demo), not evidence about a model, "
            "and must never be read as a bar. See ASSAY C1."
        )


def passes_bar(d, se):
    """PREREG BAR: margin > BAR_SIGMA * se_margin (the read beats the
    marginal predictor by more than the paired bootstrap's noise floor)."""
    _guard_not_construction_property(d, "passes_bar")
    return bool(margin_sigma(d, se) > BAR_SIGMA)


def kill_fired(d, se):
    """PREREG KILL: margin <= KILL_SIGMA * se_margin -- not even confidently
    positive at 1 SE, i.e. indistinguishable from, or worse than, a predictor
    that never looked at the input."""
    _guard_not_construction_property(d, "kill_fired")
    return bool(margin_sigma(d, se) <= KILL_SIGMA)


def format_line(d, se=None):
    """One line, printable at every eval, showing which side of I > J+KL it sits on."""
    pm = f" +/- {se['se_margin']:.4f}" if se else ""
    js = f" +/- {se['se_sharpness']:.4f}" if se else ""
    isd = f" +/- {se['se_i_q']:.4f}" if se else ""
    return (
        f"ce {d['ce']:.4f} = H {d['H_pi']:.4f} + KL {d['kl']:.4f}"
        f" + J {d['sharpness']:.4f}{js} - I {d['i_q']:.4f}{isd}"
        f" | I-(J+KL) {d['margin']:+.4f}{pm}"
        f" -> {'BEATS' if d['beats_marginal'] else 'LOSES TO'} marginal"
        f", {'beats' if d['beats_uniform'] else 'LOSES TO'} uniform"
        f" | ppl {d['ppl']:.3f} vs marg {d['ppl_marginal']:.3f} vs chance {d['ppl_chance']:.2f}"
    )


# --------------------------------------------------------------------------- #
# self-check


def _grouped_draw(n, seed):
    """A generative process whose true conditional is known by construction:
    a latent group g in {0,1,2}, each with a fixed distribution over K=4."""
    P = torch.tensor(
        [
            [0.55, 0.25, 0.15, 0.05],
            [0.05, 0.60, 0.20, 0.15],
            [0.20, 0.10, 0.45, 0.25],
        ],
        dtype=torch.float64,
    )
    g = torch.Generator().manual_seed(seed)
    grp = torch.randint(P.shape[0], (n,), generator=g)
    q = P[grp]  # the TRUE conditional: a perfectly calibrated read
    y = torch.multinomial(q, 1, generator=g).squeeze(1)
    return q, y, P


def _temper(q, T):
    """Sharpen (T<1) or flatten (T>1). Strictly monotone per row -> argmax fixed."""
    z = q.clamp_min(1e-300).pow(1.0 / T)  # 1e-300: THRESHOLD (pow-safety floor), unclassifiable under C1
    return z / z.sum(1, keepdim=True)


def _acc(q, y):
    return float((q.argmax(1) == y).to(torch.float64).mean())


if __name__ == "__main__":
    torch.manual_seed(0)

    print("(a) IDENTITY -- residual is roundoff, on random q and random y")
    for n, K, s in [(1000, 3, 1), (50000, 7, 2), (37, 2, 3), (5000, 20, 4)]:
        g = torch.Generator().manual_seed(s)
        q = torch.softmax(1.5 * torch.randn(n, K, generator=g, dtype=torch.float64), 1)
        y = torch.randint(K, (n,), generator=g)
        # CONSTRUCTION-PROPERTY: q, y are random and unrelated to any model; the
        # identity holds to roundoff regardless, which is exactly what this
        # case is asserting -- about the algebra, not about a read.
        d = decompose(q, y, K, construction_property=True)
        print(
            f"    n={n:6d} K={K:3d}  ce={d['ce']:.9f}  H={d['H_pi']:.9f}  KL={d['kl']:.9f}"
            f"  J={d['sharpness']:.9f}  I={d['i_q']:.9f}  residual={d['residual']:+.3e}"
        )
        assert abs(d["residual"]) < 1e-10  # 1e-10: THRESHOLD, unclassifiable under C1

    print()
    print("(b) CONSTANT READ -- J and I are exactly 0, ce = H(pi) + KL, hand-checkable")
    c = torch.tensor([0.50, 0.30, 0.15, 0.05], dtype=torch.float64)  # CONSTRUCTION-PROPERTY: arbitrary constant, chosen for the demo
    g = torch.Generator().manual_seed(11)
    y = torch.multinomial(torch.tensor([0.4, 0.3, 0.2, 0.1]), 20000, replacement=True, generator=g)
    q = c.expand(20000, 4).contiguous()
    # CONSTRUCTION-PROPERTY throughout this case: a constant read has J=I=0 by
    # the identity's own definition (see module docstring line 15), for ANY c
    # and ANY y -- this is a corollary of the algebra, not a fact about a model.
    d = decompose(q, y, 4, construction_property=True)
    pi = torch.bincount(y, minlength=4).to(torch.float64) / y.numel()
    hand = float(-(pi * c.log()).sum())  # sum_b pi_b (-ln c_b), by hand
    print(f"    pi        = {[round(float(v), 6) for v in pi]}")
    print(f"    c         = {[float(v) for v in c]}")
    print(f"    sharpness = {d['sharpness']:+.3e}   i_q = {d['i_q']:+.3e}")
    print(f"    ce        = {d['ce']:.12f}")
    print(f"    H+KL      = {d['H_pi'] + d['kl']:.12f}   (H={d['H_pi']:.9f}, KL={d['kl']:.9f})")
    print(f"    hand-computed sum_b pi_b(-ln c_b) = {hand:.12f}")
    assert abs(d["sharpness"]) < 1e-12 and abs(d["i_q"]) < 1e-12  # 1e-12: THRESHOLD, unclassifiable under C1
    assert abs(d["ce"] - (d["H_pi"] + d["kl"])) < 1e-12  # CONSTRUCTION-PROPERTY (corollary) checked against a THRESHOLD
    assert abs(d["ce"] - hand) < 1e-12

    print()
    print("(c) CALIBRATED READ -- q is the true conditional, must beat the marginal")
    n = 60000
    q_cal, y_c, P = _grouped_draw(n, 7)
    # CONSTRUCTION-PROPERTY, not a circularity: q_cal IS the true conditional P
    # (grp -> P[grp] -> multinomial(P[grp])), so this is the correct construction
    # of a calibrated read, and it verifies the identity -- but its numbers
    # describe the construction (an oracle, seeded and deterministic) and are
    # not evidence about any model. Reproduced from this exact seed (n=60000,
    # draw seed=7, bootstrap seed=1): margin +0.2136 +/- 0.0025 at 84.8 sigma
    # (J 0.2492 +/- 0.0005, I 0.4628 +/- 0.0026). See ASSAY C1.
    d_cal = decompose(q_cal, y_c, 4, construction_property=True)
    se_cal = bootstrap_se(q_cal, y_c, 4, n_boot=200, seed=1)
    print(f"    groups (true conditionals) = {[[float(v) for v in r] for r in P]}")
    print(f"    {format_line(d_cal, se_cal)}")
    print(f"    margin/SE = {d_cal['margin'] / se_cal['se_margin']:.1f} sigma")
    assert d_cal["beats_marginal"] and d_cal["beats_uniform"]

    print()
    print("(d) OVERCONFIDENT READ -- same argmax, same accuracy, now LOSES to the marginal")
    T = 0.3
    q_hot = _temper(q_cal, T)
    # CONSTRUCTION-PROPERTY: q_hot is a deterministic, monotone re-temperature
    # of the same oracle draw as case (c) -- still no model involved.
    d_hot = decompose(q_hot, y_c, 4, construction_property=True)
    se_hot = bootstrap_se(q_hot, y_c, 4, n_boot=200, seed=1)
    print(f"    calibrated  (T=1.0): acc = {_acc(q_cal, y_c):.6f}")
    print(f"    {format_line(d_cal, se_cal)}")
    print(f"    overconfident (T={T}): acc = {_acc(q_hot, y_c):.6f}")
    print(f"    {format_line(d_hot, se_hot)}")
    print(
        f"    argmax identical on all {n} items: {bool(torch.equal(q_cal.argmax(1), q_hot.argmax(1)))}"
        f"   accuracy delta = {_acc(q_hot, y_c) - _acc(q_cal, y_c):+.1e}"
    )
    print(
        f"    J rose {d_cal['sharpness']:.4f} -> {d_hot['sharpness']:.4f}"
        f" while I moved {d_cal['i_q']:.4f} -> {d_hot['i_q']:.4f}"
        f"; margin {d_cal['margin']:+.4f} -> {d_hot['margin']:+.4f}"
    )
    assert torch.equal(q_cal.argmax(1), q_hot.argmax(1))
    assert _acc(q_hot, y_c) == _acc(q_cal, y_c)
    assert d_cal["beats_marginal"] and not d_hot["beats_marginal"]

    print()
    print("(e) TEMPERATURE SWEEP -- the crossover, on one fixed draw and one fixed argmax")
    print(f"    {'T':>6} {'ppl':>9} {'sharpness':>11} {'i_q':>9} {'margin':>9} {'se_marg':>8} {'acc':>8}  beats_marginal")
    for T in [1.5, 1.25, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3]:
        # CONSTRUCTION-PROPERTY, every row: same oracle draw as (c)/(d), retempered
        qt = _temper(q_cal, T)
        dt = decompose(qt, y_c, 4, construction_property=True)
        st = bootstrap_se(qt, y_c, 4, n_boot=100, seed=2)
        print(
            f"    {T:6.2f} {dt['ppl']:9.4f} {dt['sharpness']:11.4f} {dt['i_q']:9.4f}"
            f" {dt['margin']:+9.4f} {st['se_margin']:8.4f} {_acc(qt, y_c):8.6f}  {dt['beats_marginal']}"
        )

    print()
    print("(f) PREREG BAR -- a positive margin is not a pass unless it clears 2 SE")
    # UNCLASSIFIABLE under ASSAY C1's two-way split: these four (margin, se)
    # pairs are hand-picked to land at specific sigma multiples -- not drawn
    # from any q,y at all, so "construction-property of a draw" does not fit;
    # not a model read either, so "measurement" does not fit. They deliberately
    # carry no construction_property tag, because this case is testing
    # passes_bar/kill_fired's own sigma-threshold logic, not making a bar claim
    # about anything -- tagging them would trip the guard below for the wrong
    # reason. See ASSAY C1's report on unclassifiable figures.
    examples = {
        "sub-1-SE (killed despite + sign)": ({"margin": 0.01}, {"se_margin": 0.02}),
        "gray (1-2 SE, neither)": ({"margin": 0.025}, {"se_margin": 0.02}),
        "clear pass (2.5 SE)": ({"margin": 0.05}, {"se_margin": 0.02}),
        "clear kill (-1.5 SE)": ({"margin": -0.03}, {"se_margin": 0.02}),
    }
    for label, (dd, ss) in examples.items():
        print(
            f"    {label:<34} sigma={margin_sigma(dd, ss):+.2f}"
            f"  passes_bar={passes_bar(dd, ss)}  kill_fired={kill_fired(dd, ss)}"
        )

    # THE PLANTED NEGATIVE this case exists to catch: sign(margin) is positive
    # for the first two rows -- exactly what bare `beats_marginal` reports --
    # and the bar must NOT pass either of them. A regression that read the bar
    # off bare sign (equivalent to BAR_SIGMA=0) would pass both; this must not.
    sub1_d, sub1_se = examples["sub-1-SE (killed despite + sign)"]
    gray_d, gray_se = examples["gray (1-2 SE, neither)"]
    win_d, win_se = examples["clear pass (2.5 SE)"]
    kill_d, kill_se = examples["clear kill (-1.5 SE)"]
    assert sub1_d["margin"] > 0 and not passes_bar(sub1_d, sub1_se) and kill_fired(sub1_d, sub1_se)
    assert gray_d["margin"] > 0 and not passes_bar(gray_d, gray_se) and not kill_fired(gray_d, gray_se)
    assert passes_bar(win_d, win_se) and not kill_fired(win_d, win_se)
    assert kill_fired(kill_d, kill_se) and not passes_bar(kill_d, kill_se)

    print()
    print("(g) GUARD -- a construction-property figure must refuse to be read as a bar")
    try:
        passes_bar(d_cal, se_cal)  # d_cal is tagged construction_property=True from case (c)
        raise AssertionError("passes_bar did not refuse a construction-property figure")
    except ConstructionPropertyAsBarError:
        pass
    try:
        kill_fired(d_cal, se_cal)
        raise AssertionError("kill_fired did not refuse a construction-property figure")
    except ConstructionPropertyAsBarError:
        pass
    print("    passes_bar/kill_fired correctly refused d_cal (construction_property=True)")

    print()
    print("all self-checks passed")
