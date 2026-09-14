"""THE NORTH-STAR RACE: one solve against depth-matched masked-attention stacks.

RED FIRST. Every assertion here was written and run before `ceqjepa/depth_race.py`
existed.

WHAT THESE BIND is the part of the race that can be wrong SILENTLY:

  - the arms are PARAMETER-MATCHED. An unmatched race measures width, not depth,
    and the confound leaves no trace in the R2 column;
  - the stack is wired to ONE HOP OF P PER LAYER, which is the assumption the
    whole depth argument rests on. It is asserted on the wiring itself: a d-layer
    arm's output must be constant in the features of any node further than d hops
    away. If that fails the depth axis is not a depth axis;
  - the reader never sees a test goal. The training goals are the 20 the lookup
    ceiling is built from, so reader and memoriser see exactly the same goals;
  - the resolvent columns q1, q2 are NOT inputs. They are hops; handing them to a
    d-layer stack silently makes it a (d+2)-layer stack;
  - the global-token arm BREAKS the one-hop assumption on purpose, so that "the
    assumption was wrong" has a measurement and not only a caveat;
  - `demo()` exits 0 (L-SURFACE: a check that cannot run is red, never skipped).
"""
import subprocess
import sys

import numpy as np
import pytest

from ceqjepa import depth_race as dr
from ceqjepa import goal_family as gf


#: `dr.race(n_test=20, n_train=10, epochs=8, seed=0)`, 62cb8e0,
#: WIN-16QAL06O9GB (python 3.11.9, torch 2.14.0+cpu), CPU only, 311.9s. The
#: TRAINED arms at the headline size cost 2,385.7s, which is not a test; this is
#: the same optimiser, the same wiring and the same draw stream at a size that
#: is. Torch's CPU path here is bit-reproducible: the 3-draw demo was run twice
#: at 62cb8e0 and every printed digit of all five trained arms agreed.
SMALL_RACE = {
    "solve": 1.0000, "trunc-1": -0.5002, "trunc-2": -0.1634,
    "trunc-4": 0.2153, "trunc-8": 0.6009,
    "attn-1": -0.7217, "attn-2": -0.6778, "attn-4": -0.6206,
    "attn-8": -0.8638, "attn-2+global": -0.6791,
}


@pytest.fixture(scope="module")
def small():
    return dr.race(n_test=3, n_train=3, epochs=2, seed=gf.SEED)


def test_the_trained_stacks_reproduce_at_a_reduced_size():
    """The optimiser is in the loop, so the trained arms need a pin of their
    own or the depth column is a number nobody re-derives."""
    r = dr.race(n_test=20, n_train=10, epochs=8, seed=gf.SEED)
    for name, w in SMALL_RACE.items():
        assert r["arms"][name]["within"] == pytest.approx(w, abs=5e-3), (
            name, r["arms"][name]["within"], w)
    assert r["one_solve_claim_retired"] is False


def test_the_advance_bound_is_never_violated(small):
    """THE CARRIED ASSUMPTION, measured. An arm limited to d hops of P can at
    best answer the draw's own mean on the rows beyond d hops, so its
    within-draw R2 cannot exceed 1 - (variance share of those rows). If any
    d-limited arm clears its own bound the assumption was wrong."""
    for name, a in small["arms"].items():
        if name == "solve":
            continue
        assert a["within"] <= a["bound_within"] + 1e-9, (name, a["within"],
                                                         a["bound_within"])
    b = [small["arms"]["trunc-%d" % d]["bound_within"] for d in dr.DEPTHS]
    assert b == sorted(b), b
    assert small["arms"]["solve"]["within"] > small["arms"]["solve"]["bound_within"],         "the bound is vacuous: the solve must exceed it, being unlimited in depth"


def test_the_arms_are_parameter_matched(small):
    """Depth is the axis. Width is not, and a race that does not say so is
    measuring width."""
    p = {a: small["arms"][a]["n_params"] for a in small["arms"]
         if a.startswith("attn-") and "global" not in a}
    assert len(p) == 4, p
    lo, hi = min(p.values()), max(p.values())
    assert hi / lo < 1.10, f"parameter counts are not matched: {p}"
    assert small["arms"]["solve"]["n_params"] <= 2, "the solve arm must be the cheapest"


def test_one_hop_of_P_per_layer_is_wired_not_assumed(small):
    """The load-bearing assumption, asserted on the network: perturbing a node
    more than d hops upstream cannot move a d-layer arm's output."""
    for d in (1, 2):
        moved = dr.receptive_field_violation(d, seed=gf.SEED)
        assert moved == 0.0, f"d={d} arm sees past {d} hops (max move {moved})"
    assert dr.receptive_field_violation(1, seed=gf.SEED, probe_hop=1) > 0.0, \
        "the receptive-field check is vacuous"


def test_the_reader_never_sees_a_test_goal(small):
    assert small["n_train"] == 3 and small["n_test"] == 3
    assert small["train_goal_ids"] and small["test_goal_ids"]
    assert not (set(small["train_goal_ids"]) & set(small["test_goal_ids"]))


def test_the_resolvents_are_not_inputs():
    assert "q1" not in dr.INPUT_NAMES and "q2" not in dr.INPUT_NAMES


def test_every_arm_is_scored_against_the_same_three_ceilings(small):
    c = small["ceilings"]
    assert set(c) == {"marginal", "lookup", "heuristic"}
    assert c["marginal"] == pytest.approx(0.0, abs=1e-12)
    for a in small["arms"].values():
        assert "within" in a and "pooled" in a and "far_within" in a


def test_the_truncation_arms_bracket_the_solve(small):
    """q_d is nested under q, so the exact-truncation arms must be ordered in d
    and must not reach the solve."""
    t = [small["arms"][f"trunc-{d}"]["within"] for d in (1, 2, 4, 8)]
    assert t[0] <= t[1] <= t[2] <= t[3] + 1e-9, t
    assert small["arms"]["solve"]["within"] > 0.999
    assert t[-1] < small["arms"]["solve"]["within"]


def test_the_global_token_arm_really_does_break_the_assumption():
    """The falsification arm. If the one-hop-per-layer wiring is what limits the
    stacks, an arm with the SAME depth and the SAME parameter count but a global
    token must reach further -- otherwise the depth axis and the range axis are
    confounded and the whole table means nothing."""
    assert dr.receptive_field_violation(2, seed=gf.SEED, probe_hop=5,
                                        use_global=True) > 0.0
    assert dr.receptive_field_violation(2, seed=gf.SEED, probe_hop=5) == 0.0
    f = len(dr.INPUT_NAMES)
    assert dr.width_for(2, f)[1] == dr.width_for(2, f, use_global=True)[1]


def test_the_exact_arms_reproduce_at_the_headline_size():
    """The half of the race that carries "computable in advance": the solve and
    the exact d-hop truncations, at the published 100 draws, with no optimiser
    anywhere in them. Re-derived here, never copied from the docstring."""
    r = dr.race(n_test=100, n_train=20, seed=gf.SEED, train_stacks=False)
    pin = dr.RESULTS
    assert r["n_rows"] == pin["n_rows"]
    for d in dr.DEPTHS:
        assert r["hop_fraction"][d] == pytest.approx(pin["hop_fraction"][d],
                                                     abs=5e-4)
    assert r["arms"]["solve"]["within"] == pytest.approx(1.0, abs=1e-9)
    for d in dr.DEPTHS:
        got = r["arms"]["trunc-%d" % d]["within"]
        assert got == pytest.approx(pin["trunc_within"][d], abs=5e-4), (d, got)
    for k in ("marginal", "lookup", "heuristic"):
        assert r["ceilings"][k] == pytest.approx(pin["ceilings"][k], abs=5e-4)
    assert r["headroom_worst"] == pytest.approx(pin["headroom_worst"], abs=5e-4)


def test_demo_exits_zero():
    p = subprocess.run(
        [sys.executable, "-m", "ceqjepa.depth_race", "--draws", "3",
         "--train", "3", "--epochs", "2"], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    assert "DEPTH TABLE" in p.stdout


def test_the_goal_pool_pins_the_unseen_draws():
    """THE SWEEP'S ONE AXIS IS THE TRAINING-GOAL COUNT, so the unseen goals must
    not move with it -- and by default they do, twice over: the training goals
    and the test goals come off ONE rng stream, and the sampler floor rejects a
    test draw against the training table, which gets stricter as that table
    grows. Comparing n_train=20 against n_train=80 without `pool` compares two
    different test sets of two different hardnesses. `pool` draws the largest
    training set, floors every test draw against ALL of it, and hands back the
    first n_train for fitting: nested training sets, one pinned test set."""
    a_tr, a_te = dr.draws(2, 2, gf.SEED, gf.FLOOR, pool=4)
    b_tr, b_te = dr.draws(2, 4, gf.SEED, gf.FLOOR, pool=4)
    assert len(a_tr) == 2 and len(b_tr) == 4
    assert all((x[0] == y[0]).all() and (x[1] == y[1]).all()
               for x, y in zip(a_tr, b_tr)), "the training sets are not nested"
    assert all((x[0] == y[0]).all() and (x[1] == y[1]).all()
               for x, y in zip(a_te, b_te)), "the unseen goals moved"
    _, c_te = dr.draws(2, 2, gf.SEED, gf.FLOOR)
    assert any((x[0] != y[0]).any() for x, y in zip(a_te, c_te)), \
        "the check is vacuous: the default path already gives the same draws"


def test_the_goal_sweep_retires_the_depth_argument():
    """THE FINDING, PINNED TO ITS OWN NUMBERS so the prose cannot drift from
    them. Every point was produced by `scratchpad/gsweep/point.py N E OUT D`
    calling `dr.race(n_test=100, n_train=N, epochs=E, seed=0, budget=9500,
    depths=(D,), with_global=False, pool=80)`; this test asserts the relations
    the conclusion rests on, not the runs, which take 90 minutes."""
    s = dr.GOAL_SWEEP
    p = s["points"]
    assert len({q["test_digest"] for q in p.values()}) == 1, "test sets moved"
    assert len({q["n_rows"] for q in p.values()}) == 1, "row counts moved"
    for q in p.values():
        assert q["pool"] == 80 and q["params"] in (9505, 9425), q
    w = {k: q["within"] for k, q in p.items()}
    # the pre-registered line, and the same line re-derived on THESE draws
    assert w["n80"] >= s["retirement_line_published"], w
    assert w["n80"] >= w["d8n20"], w
    assert w["s40e50"] >= w["d8n20"], w
    assert s["depth_argument_retired"] is True
    # monotone in goals at pinned epochs
    assert w["n20"] < w["n40"] < w["n80"], w
    # the transfer signature that was registered before the run: unseen-goal
    # fit RISES while the training-goal fit FALLS
    assert p["n80"]["train_within"] < p["n20"]["train_within"], p
    # THE INTERNAL NULL. The same training sets refit the two-parameter arms,
    # which move by less than a fortieth of what the trained stack moves.
    for d in (1, 2, 4, 8):
        moved = abs(p["n80"]["trunc_within"][d] - p["n20"]["trunc_within"][d])
        assert moved < 0.02, (d, moved)
    assert w["n80"] - w["n20"] > 0.39
