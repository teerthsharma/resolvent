"""X19: the E4' task family (join the two largest components) in M3_TASKS.

WHAT IS BEING REGISTERED. E4 -- the Rips-connectivity task of LOOP_PROMPT.md
1.7d -- was STRUCK because `_add_critical_bridge` joins the two NEAREST
components, which on S^2 is always a speck against the giant, so a static
local-degree decoder reads the label at criticality. The RULE 5 reroute (E4')
joins the two LARGEST instead. Measured, on record:

    results/e4_gate.txt:47-54  LargestJoin_S2Rips_1024 [REROUTE]
        merge 30 x 32, gate (a) closes k=32:0.0000,
        gate (b) degree-only 1.0001, +3hop 0.9951, +5hop 0.8220,
        PLANTED degree-sum 0.0000, PLANTED degree-median 0.5530.
    results/e4_gate.txt:38-45  LargestJoin_S2Rips_64 [REROUTE] still leaks:
        +5hop 0.0055 -- NOT admissible; the reroute requires n = 1024.
    STATE.md OPEN item 2: E4' passes both gates at n=1024, leak 0.1565 ->
        0.9951, and was not yet registered. This file is that registration's
        bind, written first (RED on purpose), per house law.

THE TENSOR ENCODING. One M3 example is a [s, d_model] float tensor, so the
substrate rides in the tokens: token = node, channels CH_COORD..+2 carry the
unit-sphere coordinates, channel CH_BRIDGE carries the per-example do()-bit at
the two bridge endpoints (+1 = edge present), and channels CH_QA / CH_QB mark
the queried node pair, drawn left x right across the bridge exactly as
`scale.rips_gate.draw_do_paired` draws them. Nothing about the label is stored:
`e4prime_oracle` rebuilds the Rips graph from the coordinate channels, applies
the do()-bit, and reads component membership. The label varies because the
TENSOR varies.

THE SCALE CLAUSE. Admissibility is a property of the substrate size, so the
builder refuses to draw the task below the measured floor. Registering E4'
without that clause would ship the leaking n=64 task under an admissible name.
"""
from __future__ import annotations

import functools
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceq.rips import (REROUTED_CASES, _add_critical_bridge,   # noqa: E402
                      components, rips_edges, sample_sphere)
from scale import negation_scope as NS                        # noqa: E402
from scale.rips_gate import (FAIL_BAR, adjacency, ball,       # noqa: E402
                             fit_eval, local_features,
                             planted_degree_labels, truncation_ladder)

#: The shipped admissible spec: LargestJoin_S2Rips_1024, verbatim from
#: ceq.rips.REROUTED_CASES so no seed or degree is copied by hand.
_SPEC = next(spec for spec in REROUTED_CASES if spec[1] == 1024)

D_HOUSE = 24            # the house distance default (tests/cameron/test_m3_etasks.py)
N_BATCH = 256           # drawn instances; draw_do_paired's scale is n_pairs=1024


def _require_e4prime():
    """Every test funnels through here so that RED is a plain AssertionError
    naming the missing symbol, never an AttributeError raised mid-test."""
    missing = [attr for attr in (
        "make_e4prime_batch", "e4prime_oracle", "e4prime_hop_reading",
        "e4prime_features", "e4prime_flipper_dependence")
        if not hasattr(NS, attr)]
    assert not missing, f"e4prime not implemented in scale.negation_scope: {missing}"


@functools.lru_cache(maxsize=1)
def _reference_substrate():
    """(pre_adj, post_adj, bridge_edge, left_size, right_size) rebuilt with the
    reference generator itself -- ceq.rips verbatim -- so every gate assertion
    below is tied to the MEASURED substrate, not merely to whatever the new
    builder draws."""
    _require_e4prime()
    _name, nodes, degree, seed, join = _SPEC[:5]
    points = sample_sphere(nodes, seed)
    pre_edges = rips_edges(points, degree)
    edge = _add_critical_bridge(points, pre_edges, nodes, join=join)
    assert edge is not None, "the reference draw produced no bridge"
    lab = components(nodes, pre_edges)
    sizes = sorted((sum(1 for v in lab if v == lab[c]) for c in edge),
                   reverse=True)
    return (adjacency(nodes, pre_edges), adjacency(nodes, pre_edges + [edge]),
            edge, sizes)


@functools.lru_cache(maxsize=1)
def _state():
    """The drawn tensor batch, built once for the whole module."""
    _require_e4prime()
    x, y, f, p = NS.make_e4prime_batch(N_BATCH, _SPEC[1], D_HOUSE)
    return x, y, f, p


# ------------------------------------------------------- 0. the registration --
def test_e4prime_is_registered_with_house_shape():
    """(a) An `e4prime` entry exists in M3_TASKS with the shipped 4-tuple shape
    (batch_fn, oracle_fn, feature_fn, flipper_dependence_fn) -- the shape
    `m3_capability.py` unpacks -- and the family's flipper dependence is the
    EXACT 0.0 of a label that no single token decides (that absence is
    CHECKLIST RULE 5's point), registered as a callable like every sibling's
    closed form."""
    assert "e4prime" in NS.M3_TASKS, "e4prime missing from M3_TASKS"
    _require_e4prime()
    entry = NS.M3_TASKS["e4prime"]
    assert len(entry) == 4, (len(entry),)
    batch_fn, oracle_fn, feature_fn, fd_fn = entry
    assert callable(batch_fn) and callable(oracle_fn)
    assert callable(feature_fn) and callable(fd_fn)
    assert fd_fn(_SPEC[1]) == 0.0


def test_the_label_is_recomputed_from_x_and_moves_with_its_do_bit():
    """The oracle is executable and nothing is stored: y equals
    e4prime_oracle(x, f, p) BITWISE on a drawn batch, and negating the do()-bit
    channel on a drawn subset moves exactly those labels -- the drawn-instance
    control `tests/cameron/test_m3_etasks.py` runs on every family."""
    x, y, f, p = _state()
    assert torch.equal(y, NS.e4prime_oracle(x, f, p))
    x2 = x.clone()
    flip = torch.arange(x.shape[0]) % 2 == 0
    x2[flip, :, NS.CH_BRIDGE] = -x2[flip, :, NS.CH_BRIDGE]
    moved = NS.e4prime_oracle(x2, f, p)
    assert not torch.equal(moved, y), "the do()-bit does not reach the label"
    assert torch.equal(moved[~flip], y[~flip])


def test_admissibility_scale_is_enforced_at_generation():
    """The reroute requires n = 1024 (LargestJoin_S2Rips_64 leaks 0.0055 at
    radius 5, results/e4_gate.txt:44), so the builder must RAISE below the
    measured floor rather than return an inadmissible batch under an
    admissible name, and the registered spec must land on the measured merge
    30 x 32 (results/e4_gate.txt:47)."""
    _require_e4prime()
    with pytest.raises(ValueError):
        NS.make_e4prime_batch(4, 64, D_HOUSE)
    with pytest.raises(ValueError):
        NS.make_e4prime_batch(4, 512, D_HOUSE)
    _pre, _post, _edge, sizes = _reference_substrate()
    assert sizes == [32, 30], sizes


# ------------------------------------------------- 1. gate (a), truncation ---
def test_gate_a_truncation_fires():
    """Gate (a) on the registered task, read with scale.rips_gate's own ladder
    machinery on instances read back out of the tensor markers. Measured on
    this substrate (results/e4_gate.txt:48): the k=0 rung reads sqrt(2), the
    k=1 rung is ABOVE the mean predictor (1.4128), the ladder tightens
    monotonically, and it resolves EXACTLY at the full budget (k=32:0.0000).
    A 1-hop reading near 0 would make E4' a third static task wearing an
    equilibrium's name."""
    _require_e4prime()
    pre, post, _edge, _sizes = _reference_substrate()
    x, y, _f, _p = _state()
    pairs, y01 = _pairs_and_labels(x, y)
    lad = dict(truncation_ladder(pairs, y01, [pre, post]))
    assert abs(lad[0] - math.sqrt(2.0)) < 1e-6, lad[0]
    assert lad[1] > 1.0, lad[1]
    ks = sorted(lad)
    rungs = [lad[k] for k in ks]
    assert all(a >= b - 1e-9 for a, b in zip(rungs, rungs[1:])), rungs
    assert rungs[-1] == 0.0, rungs[-1]


# ------------------------------------- 2. gate (b), decoder + planted control -
def test_gate_b_strikes_every_local_radius_and_the_planted_control_fires():
    """(c) Gate (b) on the registered task, BOTH halves.

    FAIL half: the static local decoder -- degrees and ball sizes out to
    radius 5, the strongest strictly-local features available -- stays at or
    above FAIL_BAR out to radius 3 and above 0.75 at radius 5 (measured
    1.0001 / 1.0018 / 1.0035 / 0.9951 / 0.8220, results/e4_gate.txt:49-53).
    PASS half, seen to fire on the SAME instances, features and split: the
    planted degree-sum label is read essentially exactly (< 1e-6, measured
    0.0000) and the planted balanced degree-median sits below its own linear
    floor at 0.70 (measured 0.5530), with the connectivity-vs-planted gap
    above 0.30 (measured gap 1.0001 - 0.5530 = 0.4471, CHECKLIST.md RULE 5).

    NON-DEGENERACY ON THE PASS HALF (house law after the fourteenth vacuous
    control, STATE.md item 4): the planted median label must actually vary --
    sd > 0 and neither class empty -- and the connectivity label itself must
    hold both classes, else every NRMSE above is undefined and void."""
    _require_e4prime()
    pre, post, _edge, _sizes = _reference_substrate()
    x, y, _f, _p = _state()
    pairs, y01 = _pairs_and_labels(x, y)
    graphs = [pre, post]
    # non-degeneracy of the real label
    frac = float(y01.mean())
    assert 0.0 < frac < 1.0, frac
    for radius, floor in ((0, FAIL_BAR), (1, FAIL_BAR), (2, FAIL_BAR),
                          (3, FAIL_BAR)):
        score = fit_eval(local_features(pairs, graphs, radius), y01)
        assert score >= floor, (radius, score)
    deep = fit_eval(local_features(pairs, graphs, 5), y01)
    assert deep > 0.75, deep
    # the PASS half, on identical instances / features / split
    planted = planted_degree_labels(pairs, graphs)
    x0 = local_features(pairs, graphs, 0)
    med = planted["median"]
    assert float(med.std()) > 0.0 and 0.0 < float(med.mean()) < 1.0, med
    sum_score = fit_eval(x0, planted["sum"])
    assert sum_score < 1e-6, sum_score
    median_score = fit_eval(x0, med)
    assert median_score < 0.70, median_score
    connectivity = fit_eval(x0, y01)
    assert connectivity - median_score > 0.30, (connectivity, median_score)


# ------------------------- 3. not a closed form of any bounded neighbourhood --
def test_the_label_is_not_a_closed_form_of_any_bounded_neighbourhood():
    """(d) The structural property E4' exists for, asserted the way the chain
    family asserts it (`tests/cameron/test_m3_etasks.py`, property 2) and the
    way gate (a) measured it on this substrate.

    Drawn-distribution half: the k-hop reading of the label stays at or above
    the mean predictor through k=8 (measured 1.2694 there,
    results/e4_gate.txt:48) and is still far from resolved at k=16 (measured
    0.7943), tightens monotonically, and becomes EXACTLY the label once the
    budget covers every drawn pair's hop distance. Endpoint half: on ONE drawn
    do()-on instance, taken on ITS OWN drawn pair -- left x right across the
    bridge exactly as `draw_do_paired` draws them, and never the bridge edge
    itself -- the reading says NOT-connected at one hop short of that pair's
    hop distance and connected exactly at it. No fixed neighbourhood radius
    decides the label."""
    _require_e4prime()
    pre, post, edge, _sizes = _reference_substrate()
    x, y, _f, _p = _state()
    prev = None
    for k in (0, 1, 2, 4, 8, 16, 32, 64, 128):
        got = NS.nrmse(NS.e4prime_hop_reading(x, _f, _p, k), y)
        assert got == got, (k, got)          # NaN would be a vacuous control
        if prev is not None:
            assert got <= prev + 1e-9, (k, got, prev)
        if k <= 8:
            assert got >= 1.0, (k, got)
        if k == 16:
            assert got > 0.75, (k, got)
        prev = got
    assert prev == 0.0, prev
    # endpoint half: ONE drawn do()-on instance, on ITS OWN drawn pair. The
    # bridge edge itself is the one cross-component pair that IS adjacent in
    # the post graph -- it is the added edge (scale/negation_scope.py:691,
    # ceq/rips.py:204) -- so re-marking to its endpoints would make `dist > 1`
    # impossible for every faithful implementation and the near/exact demo
    # vacuous (k=0 vs k=1). Every OTHER left-x-right pair is non-adjacent by
    # construction: the pre graph has no cross edges and the post graph adds
    # only this one, which is why the control below can demand dist > 1.
    on = int((y > 0).nonzero()[0].item())
    xe = x[on:on + 1]
    qi = int((xe[0, :, NS.CH_QA] != 0).nonzero()[0].item())
    qj = int((xe[0, :, NS.CH_QB] != 0).nonzero()[0].item())
    assert (qi, qj) != tuple(edge), "the draw handed the control the bridge"
    dist = _hop_distance(post, qi, qj)
    assert dist > 1, dist
    near = NS.e4prime_hop_reading(xe, _f, _p, dist - 1)
    exact = NS.e4prime_hop_reading(xe, _f, _p, dist)
    assert float(near[0]) == -1.0, float(near[0])
    assert float(exact[0]) == 1.0, float(exact[0])


# ------------------------------------------------------------------ helpers --
def _pairs_and_labels(x, y):
    """((i, j), which_graph) instances read back OUT of the tensor markers, in
    the format scale.rips_gate's decoders consume, with the label in that
    machinery's 0/1 convention. Reading the pairs back from the markers rather
    than from builder internals makes this a round-trip check of the encoding.
    """
    n, s, _c = x.shape
    qa = (x[:, :, NS.CH_QA] != 0)
    qb = (x[:, :, NS.CH_QB] != 0)
    br = (x[:, :, NS.CH_BRIDGE] != 0)
    assert int(qa.sum()) == n and int(qb.sum()) == n, "markers malformed"
    assert all(int(br[b].sum()) == 2 for b in range(n)), "bridge markers malformed"
    i_idx = qa.nonzero()[:, 1].tolist()
    j_idx = qb.nonzero()[:, 1].tolist()
    bits = [1 if float(x[b, br[b].nonzero()[0].item(), NS.CH_BRIDGE]) > 0 else 0
            for b in range(n)]
    pairs = [((i_idx[t], j_idx[t]), bits[t]) for t in range(n)]
    y01 = (y > 0).to(torch.float32).numpy()
    assert sorted(set(y01.tolist())) == [0.0, 1.0], "label degenerate"
    return pairs, y01


def _hop_distance(adj, src, dst):
    """Smallest r with dst inside src's r-ball, via rips_gate.ball."""
    for r in range(1, len(adj)):
        if dst in ball(adj, src, r):
            return r
    raise AssertionError("unreachable")
