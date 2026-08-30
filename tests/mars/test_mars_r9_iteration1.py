"""MARS / MORIARTY, R9 iteration 1 -- the standing adversary's battery.

CHARACTERISATION, NOT REPAIR. Every assertion below records a defect that is
live in the tree at 74e5590. Each test passes today because the defect is
present; each would fail if the defect were fixed. Nothing here repairs
anything, and nothing here touches an existing test.

Cross-references: FINDINGS.md A1/A4/A5/A6 name four IMPACT defects already. The
seven below are additional, and the first of them is of a different order: it
says the registered task cannot be learned across the split the harness draws.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scale.impact import (CH_A_DEG, CH_NEWS, _fit_eval, _impact_query_node,
                          _nrmse, build_impact_graph, impact_covariates,
                          impact_features, impact_planted_features,
                          impact_sign_gate, make_impact_batch)

S = 1024


@pytest.fixture(scope="module")
def graph0():
    return build_impact_graph(S, 0)


@pytest.fixture(scope="module")
def batch0():
    return make_impact_batch(512, S, 24, seed=0)


# ------------------------------------------------------------------ ATTACK 1 --
def test_impact_label_does_not_transfer_across_the_harness_held_out_seed():
    """IMPACT is exactly linear WITHIN a graph and uncorrelated ACROSS graphs.

    `scale/m3_quintuple.py:478-480` draws train at `seed` and eval at
    `seed + 12345`; `scale/impact.py:566` sets `graph_seed = seed % 1000000`, so
    the eval batch sits on a DIFFERENT graph. The label is `y = w . news` with
    `w = B^T K^T e_q` fixed per graph, and neither `A` nor `B` is encoded in the
    batch tensor -- channels 2..9 are reserved and unused, so `x` carries only
    news, degree, rho and the seed scalar.

    Consequence, asserted here on both halves: a plain linear probe over every
    news and degree coordinate reads the train graph to machine zero, and the
    same probe transferred to the eval graph reads WORSE than the mean
    predictor. Any arm trained on this task is scored on a functional drawn
    independently of everything it saw.
    """
    def feats(x):
        n = x.shape[0]
        return np.concatenate([np.ones((n, 1)),
                               x[:, :, CH_NEWS].numpy(),
                               x[:, :, CH_A_DEG].numpy()], axis=1)

    xt, yt, _, _ = make_impact_batch(2560, S, 24, seed=0)
    xe, ye, _, _ = make_impact_batch(512, S, 24, seed=0 + 12345)
    Xt, Xe = feats(xt), feats(xe)
    yt, ye = yt.numpy(), ye.numpy()

    w = np.linalg.solve(Xt[:2048].T @ Xt[:2048] + 1e-6 * np.eye(Xt.shape[1]),
                        Xt[:2048].T @ yt[:2048])
    within = _nrmse(Xt[2048:] @ w, yt[2048:])
    across = _nrmse(Xe @ w, ye)

    assert within < 1e-4, (
        f"planted positive failed: the task should be exactly linear within a "
        f"graph, got held-out NRMSE {within}")
    assert across > 1.0, (
        f"transfer NRMSE {across} -- expected above the mean predictor's 1.0")

    # Second, independent path: closed form, no fitting.
    g0, g1 = build_impact_graph(S, 0), build_impact_graph(S, 12345)
    e0 = np.zeros(S)
    e0[_impact_query_node(g0)] = 1.0
    e1 = np.zeros(S)
    e1[_impact_query_node(g1)] = 1.0
    w0, w1 = g0.B.T @ (g0.K.T @ e0), g1.B.T @ (g1.K.T @ e1)
    cos = float(w0 @ w1 / (np.linalg.norm(w0) * np.linalg.norm(w1)))
    assert abs(cos) < 1e-3, f"train/eval label functionals are aligned: cos={cos}"


# ------------------------------------------------------------------ ATTACK 2 --
def test_impact_sign_gate_fires_on_a_perturbation_with_zero_sign_flips(graph0, batch0):
    """Gate 3 is not specific to sign. `scale/impact.py:948` passes on
    `degrade_aware > 0.10 and ci_lo > 0.0`, and `X_plant_aware` is built from
    the PLANTED `B` while the label is rebuilt from a perturbed `B` -- so ANY
    perturbation of `B` drives the features off the label.

    Here `B`'s magnitudes are redrawn from the same U[0.5, 1] and every sign is
    left exactly as planted. The gate's threshold is crossed anyway at seed 0
    (0.1831) and at seed 2 (0.1452); at seed 1 it reads 0.0701 and does not
    cross, so the failure is 2 of the 3 seeds measured, not all of them.
    """
    x, y, f, p = batch0
    yn = y.numpy()
    q = _impact_query_node(graph0)
    news = x[:, :, CH_NEWS].numpy().astype(np.float64)
    Xa = impact_planted_features(x, f, p).numpy()
    base = _fit_eval(Xa, yn)

    mask = graph0.B != 0
    rng = np.random.RandomState(7)
    b_mag = graph0.B.copy()
    b_mag[mask] = np.sign(b_mag[mask]) * (0.5 + 0.5 * rng.rand(int(mask.sum())))
    assert int((np.sign(b_mag) != np.sign(graph0.B)).sum()) == 0

    y_mag = (graph0.K @ (b_mag @ news.T)).T[:, q]
    degrade = _fit_eval(Xa, y_mag) - base
    assert degrade > 0.10, (
        f"expected the sign gate to be crossed by a sign-preserving "
        f"perturbation at seed 0, got degrade {degrade}")


def test_impact_sign_gate_local_must_fire_is_negative_and_is_not_in_the_verdict():
    """`scale/impact.py:872` -- "sign-scrambled B must degrade EVERY arm's
    score". The local arm's score IMPROVES under scrambling, and `fires` at
    `:948` reads only `degrade_aware`, so the half that does not fire is
    computed, returned, printed by `impact_report`, and excluded from PASS.
    """
    r = impact_sign_gate(n=512, s=S, seed=0)
    assert r["degrade_local"] < 0.0, r["degrade_local"]
    assert r["fires"] is True, "the gate passes anyway"


# ------------------------------------------------------------------ ATTACK 3 --
def test_impact_local_decoder_features_are_rank_two_not_four(batch0):
    """`impact_features` returns `[1, news_q, deg_q, news_q*deg_q]`, but the
    graph is fixed for the whole batch, so `deg_q` is one scalar repeated down
    the column. Columns 2 and 3 are exact multiples of columns 0 and 1: the
    gate's FAIL probe is a two-parameter fit dressed as a four-parameter one,
    and the ridge `1e-6` is what makes the singular normal matrix solvable.
    """
    x, y, f, p = batch0
    mat = impact_features(x, f, p).numpy()
    assert mat.shape[1] == 4
    assert np.linalg.matrix_rank(mat) == 2
    assert len(np.unique(mat[:, 2])) == 1
    assert np.array_equal(mat[:, 3], mat[:, 1] * mat[0, 2])
    # the two dependent columns move the gate's headline number by ~4e-10
    assert abs(_fit_eval(mat[:, :2], y.numpy()) - _fit_eval(mat, y.numpy())) < 1e-8


# ------------------------------------------------------------------ ATTACK 4 --
def test_impact_label_frac_non_degeneracy_check_is_an_identity(batch0):
    """`0.0 < frac_median < 1.0` (`scale/impact.py:816`, `:922`) is installed as
    the rule-4 non-degeneracy guard. `frac_median` is the fraction of a sample
    STRICTLY ABOVE ITS OWN MEDIAN, which is 0.5 for any even-sized continuous
    sample -- including a label whose sd is exactly 0.0. The guard cannot
    discriminate; only the `label_sd > 1e-6` conjunct in the same expression
    does any work.
    """
    def frac(v):
        return float((v > np.median(v)).mean())

    _, y, _, _ = batch0
    rng = np.random.RandomState(0)
    degenerate = y.numpy() * 1e-30
    assert float(degenerate.std()) == 0.0
    for sample in (y.numpy(), rng.randn(512), degenerate):
        assert frac(sample) == 0.5
        assert 0.0 < frac(sample) < 1.0          # the shipped guard, unfalsifiable


# ------------------------------------------------------------------ ATTACK 5 --
@pytest.mark.parametrize("seed", [0, 1, 12345])
def test_impact_spectral_radius_covariate_is_a_construction_constant(seed):
    """Gate 4 is "covariates printed PER INSTANCE". `scale/impact.py:311` sets
    `rho = RHO_TARGET / spectral_radius_raw` and `impact_covariates` then
    reports `rho * spectral_radius_raw`, which is `RHO_TARGET` identically. The
    Hawkes proxy inherits it: `0.88 * 0.90 = 0.792` is 98.3% of its value at
    seed 0. `impact_report` also writes Gate 4's verdict as the literal string
    "=> PASS" and leaves it out of the `all_pass` conjunction entirely.
    """
    cov = impact_covariates(build_impact_graph(S, seed))
    assert cov["spectral_radius"] == 0.88
    assert 0.79 < cov["hawkes_branching"] < 0.81


# ------------------------------------------------------------------ ATTACK 6 --
def test_argmax_cell_has_no_autograd_path_to_the_pivot_gate():
    """`scale/m3_quintuple.py:295-297` builds the argmax alpha as
    `torch.zeros_like(log_gate).scatter_(...)`, which carries no grad_fn. So the
    `argmax` cell's choice of WHICH pivot to read is never trained, while
    `twin` and `settled` train theirs.

    That makes `argmax - softmax = -0.118456` a contrast between softmax and
    "read one pivot chosen by an UNTRAINED gate" -- not between mixture and
    lookup. The reading at CHECKLIST.md:1206 ("reading one pivot is worse than
    reading none. The MIXTURE is the whole contribution") is confounded with
    trained-versus-untrained selection.
    """
    import scale.m3_quintuple as mq
    import scale.negation_scope as ns

    x, _y, _f, _p = ns.make_batch(64, 64, 24, d_model=16, seed=0)
    grads = {}
    for cell in ("twin", "settled", "argmax"):
        torch.manual_seed(0)
        arm = mq.QuintArm("softmax", 64, cell=cell, k_piv=8)
        assert arm.base_cell == cell
        q, kk = arm.wq(x), arm.wk(x)
        piv = mq.batched_pivots(kk, 8)
        log_gram, log_gate, _av = mq.batched_log_pivot_context(
            q, kk, x, piv, need_gram=(cell == "settled"))
        log_gate = log_gate.detach().requires_grad_(True)
        if cell == "settled":
            a = mq.BatchedSettled.apply(log_gate, log_gram.detach(),
                                        0.5, 21, 21).exp()
        elif cell == "twin":
            a = (log_gate - torch.logsumexp(log_gate, -1, keepdim=True)).exp()
        else:
            a = torch.zeros_like(log_gate)
            a.scatter_(1, log_gate.argmax(-1, keepdim=True), 1.0)
        if not a.requires_grad:
            grads[cell] = 0.0
            continue
        w = torch.randn_like(a)
        grads[cell] = float(
            torch.autograd.grad((a * w).sum(), log_gate)[0].abs().sum())

    assert grads["argmax"] == 0.0
    assert grads["twin"] > 1.0 and grads["settled"] > 1.0, grads


# ------------------------------------------------------------------ ATTACK 7 --
def test_rips_counts_survive_the_independent_derivation_its_header_promises():
    """`ceq/rips.py:28-30` says the component counts are "checked against an
    independent derivation rather than assumed to transfer". No such derivation
    is in the tree: `_count` (`:154`) calls `components` (`:110`), and every
    test asserts against numbers that same function produced.

    This supplies one. A union-find count agrees 6/6, and the smallest gap
    between any pair's dot product and `cos(r)` is 7.16e-07 -- nine orders of
    magnitude above the ~1e-16 a 1-ULP libm disagreement could move it, so the
    counts do transfer and the header's caveat is idle. The MISSING piece was
    the derivation, not the fidelity.
    """
    from ceq.rips import CASES, make_case, sample_sphere

    def union_find_count(node_count, edges):
        par = list(range(node_count))
        active = [False] * node_count

        def find(v):
            while par[v] != v:
                par[v] = par[par[v]]
                v = par[v]
            return v

        for a, b in edges:
            active[a] = active[b] = True
            if a != b:
                ra, rb = find(a), find(b)
                if ra != rb:
                    par[ra] = rb
        return len({find(v) for v in range(node_count) if active[v]})

    counts, worst = [], math.inf
    for spec in CASES:
        case = make_case(*spec)
        assert union_find_count(case.node_count, case.edges) == case.n_components
        counts.append(case.n_components)
        pts = sample_sphere(spec[1], spec[3])
        minimum_dot = math.cos(2.0 * math.asin(math.sqrt(spec[2] / (spec[1] - 1))))
        for i in range(spec[1]):
            xi, yi, zi = pts[i]
            for j in range(i + 1, spec[1]):
                xj, yj, zj = pts[j]
                gap = abs(xi * xj + yi * yj + zi * zj - minimum_dot)
                worst = min(worst, gap)
    assert counts == [15, 6, 1, 1, 178, 3], counts
    assert worst > 1e-9, worst
