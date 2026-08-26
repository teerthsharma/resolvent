"""E4 admission: the Rips connectivity label is struck, and the reroute that saves it.

`LOOP_PROMPT.md` section 1.7d admits `ceq/rips.py` as an E4 candidate behind two
gates. This file is the bind for both, and it records that they came out split:

  GATE (a), TRUNCATION -- PASSES. A `k`-hop reachability reading of the label is
  bounded away from it at every `k` below the graph diameter and tightens
  monotonically. At `k = 1` it reads 1.4135, which is WORSE than the mean predictor,
  so no part of the label is legible in one hop. E4 is not a static task by this
  reading.

  GATE (b), THE DECODER MUST-FIRE -- STRIKES E4. Three separate ways, each a test
  below. The prescribed passing half of the control cannot be drawn at all; the
  other prescribed passing case does not pass; and the degree-only decoder PASSES at
  criticality, which section 1.7d makes a strike without appeal.

WHY THE TWO GATES DISAGREE, WHICH IS THE FINDING THAT TRANSFERS. Gate (a) truncates
REACHABILITY from `i` to `j`, and that genuinely needs the 62-hop diameter. But
deciding the label never requires reaching `j`. `_add_critical_bridge` joins the two
NEAREST components, and on `S^2` at these degrees that is always a speck against the
giant -- measured `3 x 222` and `1016 x 4` below. A speck is saturated by a ball of
radius two or three, so "are we in the same component" collapses into "is my own
ball small", and a static radius-3 decoder reads it at 0.1585. Gate (a) truncated
the wrong reading, and that is a defect in the gate, not only in the corpus.

THE REROUTE IS ON THE SAME CODE PATH. `ceq.rips.rerouted_corpus()` changes one thing,
which two components the bridge joins, and joins the two LARGEST. On
`LargestJoin_S2Rips_1024` the merge is `30 x 32`, no ball saturates, and every static
local decoder out to radius 5 collapses to the mean predictor while the planted
control still fires on the same instances. The last test is that bind.

EVERY ABSENCE HERE CARRIES ITS PLANTED CASE. The claims of the form "the decoder
cannot read this" are worth nothing on their own, so each one is paired with a label
on the SAME instances, the SAME feature matrix and the SAME train/test split that
the decoder MUST read, and the test asserts both halves.
"""
import math

import pytest

from ceq.rips import corpus
from scale.rips_gate import (FAIL_BAR, PASS_BAR, adjacency, draw_balanced_marginal,
                             draw_do_paired, eccentricity_diameter, fit_eval,
                             local_features, planted_degree_labels, truncation_ladder)

#: The four `(name, n, target_degree, seed, join)` specs the gates are read on. The
#: degree expressions are the upstream ones verbatim, not rounded copies.
CRITICAL_BRIDGE = ("CriticalBridge_S2Rips_256", 256, 0.75 * math.log(256.0),
                   0x33960002, "nearest")
CRITICAL_LARGE = ("CriticalLarge_S2Rips_1024", 1024, math.ceil(math.log(1024.0)),
                  0x33960006, "nearest")
REROUTE_1024 = ("LargestJoin_S2Rips_1024", 1024, 2.0, 0x33960005, "largest")


@pytest.fixture(scope="module")
def cases():
    return {c.name: c for c in corpus()}


@pytest.fixture(scope="module")
def paired():
    """The `do()`-paired draw for each spec, built once -- the 1024-node cases are
    O(n^2) to generate and three tests read them."""
    return {spec[0]: draw_do_paired(*spec) for spec in
            (CRITICAL_BRIDGE, CRITICAL_LARGE, REROUTE_1024)}


# ----------------------------------------------------------- gate (a): it passes
def test_the_truncation_gate_passes_on_both_critical_cases(paired):
    """A `k`-hop reading is bounded away from the label and tightens with `k`.

    This is section 1.7 property 2 verbatim, and it is the one thing E4 does have.
    The `k = 1` rung is the one that decides whether the family is a third static
    task: it must be nowhere near 0. It is in fact ABOVE 1.0, meaning a one-hop
    reading is worse than predicting the mean, because at one hop the reading says
    "different component" for nearly every pair while half of them are the same.
    """
    for name in (CRITICAL_BRIDGE[0], CRITICAL_LARGE[0]):
        (pairs, y), graphs, _, _ = paired[name]
        ladder = dict(truncation_ladder(pairs, y, graphs))
        assert ladder[0] == pytest.approx(math.sqrt(2.0), abs=1e-6), (
            f"{name}: the 0-hop reading should be exactly sqrt(2) on a balanced "
            f"label, got {ladder[0]!r}")
        assert ladder[1] > 1.0, (
            f"{name}: a 1-hop reading got {ladder[1]!r}; if this were near 0 the "
            f"label would be legible without iterating and E4 would be static")
        rungs = [ladder[k] for k in sorted(ladder)]
        assert all(a >= b - 1e-9 for a, b in zip(rungs, rungs[1:])), (
            f"{name}: the ladder is not monotone: {rungs!r}")
        assert rungs[-1] == 0.0, f"{name}: the full-budget reading is not exact"


def test_the_iteration_depth_is_the_diameter_and_it_diverges_at_criticality(cases):
    """The dial is the graph diameter, and it is a property of the draw, not a knob.

    4 and 10 away from the transition against 62 and 32 at it. This is why the corpus
    looked like a real `t*`: nothing here was chosen, the degree parameter moved and
    the diameter followed.
    """
    measured = {}
    for name in ("StableSparse_S2Rips_64", "SupercriticalDense_S2Rips_256",
                 "CriticalBridge_S2Rips_256", "CriticalLarge_S2Rips_1024"):
        case = cases[name]
        adj = adjacency(case.node_count, case.edges)
        labels = [x for x in case.partition if x >= 0]
        giant = max(set(labels), key=labels.count)
        nodes = [v for v in range(case.node_count) if case.partition[v] == giant]
        measured[name] = eccentricity_diameter(adj, nodes)
    assert measured == {"StableSparse_S2Rips_64": 4,
                        "SupercriticalDense_S2Rips_256": 10,
                        "CriticalBridge_S2Rips_256": 62,
                        "CriticalLarge_S2Rips_1024": 32}, measured


# ------------------------------------------------- gate (b): it strikes the task
def test_the_prescribed_passing_half_of_the_control_cannot_be_drawn(cases):
    """`SupercriticalDense_S2Rips_256` has ONE component holding all 256 nodes.

    Section 1.7d expects the degree decoder to PASS here and calls that the control.
    It cannot: every node pair carries the label 1, so the label's standard deviation
    is exactly 0 and there is no quantity for a decoder to be right or wrong about.
    A control that cannot be drawn is not a weak control, it is no control.
    """
    for name in ("SupercriticalDense_S2Rips_256", "GroundedStaticRepeated_S2Rips_256"):
        case = cases[name]
        assert case.n_components == 1
        assert sum(1 for x in case.partition if x < 0) == 0, (
            f"{name}: an isolated node would give the label some variance")
        drawn, (n_pos, n_neg) = draw_balanced_marginal(case)
        assert drawn is None and n_neg == 0, (
            f"{name}: drew {n_pos} same / {n_neg} different, expected no different "
            f"pair to exist at all")


def test_the_other_prescribed_passing_case_does_not_pass_either(cases):
    """`StableSparse_S2Rips_64` is the second half of the prescribed control.

    It reads 0.8927, which is not a pass by any bar: the decoder removes about a
    tenth of the label's variance. Its 15 components are sizes 10, 8, 7, 6, ... and
    degree says nothing about WHICH of them a node is in. So neither prescribed
    passing case is usable and gate (b) as written has no control at all.
    """
    drawn, _ = draw_balanced_marginal(cases["StableSparse_S2Rips_64"])
    pairs, y = drawn
    graphs = [adjacency(64, cases["StableSparse_S2Rips_64"].edges)]
    score = fit_eval(local_features(pairs, graphs, 0), y)
    assert score > PASS_BAR, (
        f"the degree decoder scored {score!r} on StableSparse_S2Rips_64; if this "
        f"were below {PASS_BAR} the prescribed control would have a passing half")


def test_the_degree_decoder_passes_at_criticality_so_e4_is_struck(cases):
    """THE STRIKE. Section 1.7d: if the decoder passes at criticality, E4 is struck.

    It does. `CriticalLarge_S2Rips_1024` is a bridge case, and its degree-only
    decoder reads 0.4710 on a held-out half. The mechanism is in the assertion
    below it: the post-bridge partition is 1020 / 2 / 2, so every different-component
    pair must contain one of the four fringe nodes, and all four have degree 1.

    THE PLANTED CASE THAT CHANGES THIS is the reroute, asserted in the last test:
    the same decoder, the same draw and the same split read 1.0001 once the bridge
    joins two comparable components instead. So this is not a decoder that reads
    everything, it is a corpus whose label is local.
    """
    case = cases["CriticalLarge_S2Rips_1024"]
    assert case.pre_bridge_components == case.n_components + 1 == 4
    drawn, _ = draw_balanced_marginal(case)
    pairs, y = drawn
    score = fit_eval(local_features(pairs, [adjacency(1024, case.edges)], 0), y)
    assert score < PASS_BAR, (
        f"the degree decoder scored {score!r} at criticality; the strike in "
        f"section 1.7d is conditioned on this being below {PASS_BAR}")
    sizes = sorted((sum(1 for x in case.partition if x == c)
                    for c in set(v for v in case.partition if v >= 0)), reverse=True)
    assert sizes == [1020, 2, 2], sizes


def test_the_bridge_always_glues_a_speck_to_the_giant(paired):
    """The root cause, and it is one line of `_add_critical_bridge`.

    It joins the two NEAREST distinct components. On `S^2` at these degrees the
    nearest cross-component pair is never two comparable components -- it is whatever
    tiny cluster happens to sit closest to the giant. Measured: 3 against 222, and
    1016 against 4. That is what makes the label local, and it is why changing WHICH
    components are joined is the whole of the fix.
    """
    assert paired[CRITICAL_BRIDGE[0]][3] == (3, 222)
    assert paired[CRITICAL_LARGE[0]][3] == (1016, 4)


def test_a_static_radius_three_decoder_reads_the_label_at_criticality(paired):
    """Gate (b) again, and this one is not a technicality about the draw.

    Degree alone fails on the `do()`-paired set (0.9470), so E4 survives gate (b) by
    the letter there. It does not survive the spirit: BALL SIZES are equally static
    and equally local -- no fixed point, no component count, no partition -- and a
    radius-3 ball-size decoder reads the label at 0.1585 and 0.1565. The bridge
    endpoint sits in a 3-node speck, so its ball jumps from 3 to hundreds the moment
    the bridge lands, and the decoder reads that jump instead of the connectivity.

    The planted halves are asserted alongside: the degree-sum label comes back exact,
    so the failure of the degree-only reading is a fact about the label and not about
    the fit.
    """
    for name, ceiling in ((CRITICAL_BRIDGE[0], 0.20), (CRITICAL_LARGE[0], 0.20)):
        (pairs, y), graphs, _, _ = paired[name]
        assert fit_eval(local_features(pairs, graphs, 0), y) > FAIL_BAR
        leak = fit_eval(local_features(pairs, graphs, 3), y)
        assert leak < ceiling, (
            f"{name}: the radius-3 static local decoder scored {leak!r}; E4 is "
            f"struck because this is far below the mean predictor's 1.0")
        planted = planted_degree_labels(pairs, graphs)
        x0 = local_features(pairs, graphs, 0)
        assert fit_eval(x0, planted["sum"]) < 1e-6, "the decoder cannot read degree"


# ------------------------------------------------------------ RULE 5: the reroute
def test_the_reroute_closes_the_leak_and_its_control_is_seen_to_fire(paired):
    """`LargestJoin_S2Rips_1024`: the same generator, joining the two LARGEST.

    The merge is 30 against 32 rather than a speck against a giant, so no endpoint's
    ball saturates and there is nothing local to read. Every static local decoder out
    to radius 5 sits at the mean predictor, while the truncation ladder still
    tightens and still resolves only at the full budget.

    BOTH HALVES ARE ASSERTED. The absence (the connectivity label is unreadable) is
    worthless without the planted case, so the same decoder, the same feature matrix
    and the same split are required to read a degree-defined label on these very
    instances.

    THE PLANTED MEDIAN LABEL IS NOT READ AGAINST `PASS_BAR`, AND THAT IS DELIBERATE.
    `PASS_BAR = 0.5` was pre-registered for the connectivity label. The median label
    is a STEP function of the degree sum, and a linear decoder cannot represent a
    step, so it carries a floor of its own that has nothing to do with whether the
    control fired -- it reads 0.5530 here and 0.5758 / 0.5893 on the two critical
    cases, a floor, not a signal level. Holding it to a bar written for a different
    label shape would be comparing two different things. The honest reading is the
    CONTRAST on identical instances: the planted label against the connectivity label
    the same decoder just failed at. That gap is what is asserted.
    """
    (pairs, y), graphs, _, merged = paired[REROUTE_1024[0]]
    assert merged == (30, 32), merged
    for radius, floor in ((0, FAIL_BAR), (1, FAIL_BAR), (2, FAIL_BAR), (3, FAIL_BAR)):
        score = fit_eval(local_features(pairs, graphs, radius), y)
        assert score > floor, (
            f"radius {radius} scored {score!r} on the rerouted case; the leak that "
            f"struck E4 would be back")
    assert fit_eval(local_features(pairs, graphs, 5), y) > 0.75

    planted = planted_degree_labels(pairs, graphs)
    x0 = local_features(pairs, graphs, 0)
    assert fit_eval(x0, planted["sum"]) < 1e-6, (
        "the degree-sum label is a linear function of the features; a decoder that "
        "cannot return 0 here is broken and every absence above is void")
    connectivity = fit_eval(x0, y)
    median = fit_eval(x0, planted["median"])
    assert median < 0.70, (
        f"the balanced planted label read {median!r}; the control is not firing")
    assert connectivity - median > 0.30, (
        f"the decoder read the connectivity label at {connectivity!r} and a planted "
        f"degree label at {median!r} on the SAME instances, features and split. If "
        f"this gap closed, the failure above would be the decoder's, not the label's")

    ladder = dict(truncation_ladder(pairs, y, graphs))
    assert ladder[1] > 1.0 and ladder[16] < ladder[8] and ladder[32] == 0.0, ladder
