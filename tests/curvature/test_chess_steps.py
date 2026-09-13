"""The chess bed: an EXACT plies-to-loss oracle, and T-STEPS against its controls.

RED FIRST. Every test here was written and run before ceqjepa/chess_steps.py
existed, and the verbatim failure is recorded on the board.

WHAT IS BEING PINNED. The owner's spec asked for an engine oracle and Wilson
established there is no engine on this box. The spec's other door is
ENUMERATION, and on K+Q vs K the legal position space is small enough to solve
completely by backward induction: distance-to-mate is then the ground truth an
engine approximates, not an approximation of an engine. These tests pin that the
oracle is exact (no depth cap, no truncation, no zeros standing in for draws),
that the T-STEPS number is never reported without its Euclidean control and its
unweighted-graph arm, and that the planted negatives fire.
"""

import re
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from ceqjepa import chess_steps as cs
from ceqjepa.curvature import ALPHA, K_NEIGHBORS, is_refusal

MODULE = Path(cs.__file__)


# ---------------------------------------------------------------------------
# 1. THE EXACT ORACLE
# ---------------------------------------------------------------------------

def test_the_space_is_the_whole_legal_space_and_is_closed_under_moves():
    """Every successor of an enumerated position is itself enumerated.

    An oracle over a space that is not closed is a search with a silent depth
    cap wearing a different name.
    """
    sp = cs.space()
    assert sp["n_positions"] > 100_000, sp["n_positions"]
    assert sp["n_edges"] > 1_000_000, sp["n_edges"]
    succ = sp["succ"]
    assert succ.min() >= 0
    # the space carries ONE extra state: K vs K, where the queen's capture leads.
    # Without it the space would not be closed and the oracle would be a search.
    assert sp["n_states"] == sp["n_positions"] + 1
    assert succ.max() < sp["n_states"]
    # the indptr is a real CSR over every position
    assert sp["succ_indptr"].shape[0] == sp["n_positions"] + 1
    assert int(sp["succ_indptr"][-1]) == sp["n_edges"]


def test_known_mate_in_n_reads_exactly_n():
    """Positions whose distance to mate is known independently of this code."""
    # checkmate on the board: Black to move, mated. 0 plies.
    assert cs.dtm("4k3/4Q3/4K3/8/8/8/8/8 b - - 0 1") == 0
    # White to move with Qe7# available. 1 ply.
    assert cs.dtm("4k3/8/4K3/4Q3/8/8/8/8 w - - 0 1") == 1
    # Black to move, every legal reply allows mate next ply. 2 plies. Established
    # by exhaustive python-chess search over the whole space, in a script that
    # shares no code with the module (recorded on the board).
    assert cs.dtm("8/8/8/8/8/8/8/k1KQ4 b - - 0 1") == 2


def test_the_published_maximum_for_this_endgame_is_reproduced():
    """K+Q vs K is a win in at most 10 moves. Ten white moves and the nine black
    replies between them is 19 plies, counted from the position White faces. It
    is an external fact this code did not produce; the oracle must land on it
    exactly, and one ply either side is a bug."""
    o = cs.oracle()
    assert o["max_dtm_white_to_move"] == cs.KQK_MAX_DTM_PLIES == 19
    # the defender's slice runs one ply longer, and nothing runs longer than that
    assert o["max_dtm_black_to_move"] == 20


def test_a_second_algorithm_agrees_on_every_shallow_position():
    """Forward alpha-beta minimax, written separately from the retrograde
    induction, must reproduce the same distance-to-mate. Two algorithms, one
    number: the check the exact-W1 round taught this project to demand."""
    worst, n = cs.cross_check_forward_search(max_plies=5, n_sample=300, seed=0)
    assert n >= 300, n
    assert worst == 0, "forward search disagrees on %d positions" % worst


def test_draws_are_refusals_with_reasons_never_zeros():
    """A stalemate is not a mate in zero. Each drawn cause carries its own
    reason, so a caller can count them, and a threshold applied to one fails
    loudly instead of passing the way `0 <= tol` silently does."""
    stalemate = cs.dtm("k7/2Q5/8/8/8/8/8/K7 b - - 0 1")
    assert is_refusal(stalemate), stalemate
    assert "stalemate" in stalemate.reason.lower()
    assert not isinstance(stalemate, (int, float, np.integer, np.floating))

    o = cs.oracle()
    assert o["n_draw"] > 0
    codes = set(o["refusals"])
    assert len(codes) >= 2, codes


def test_the_oracle_has_no_depth_cap():
    """Nothing in the source truncates the induction, and the fixpoint is
    reached: no position is left labelled 'unresolved'."""
    src = MODULE.read_text(encoding="utf-8")
    assert not re.search(r"max_depth\s*=\s*\d+", src), "a depth cap is in the source"
    o = cs.oracle()
    assert o["n_unresolved"] == 0, o["n_unresolved"]
    assert o["n_win"] + o["n_draw"] == cs.space()["n_positions"]


def test_the_committor_is_exact_and_pinned_at_both_boundaries():
    o = cs.oracle()
    q = o["q"]
    assert q.shape[0] == cs.space()["n_positions"]
    assert float(q.min()) >= 0.0 and float(q.max()) <= 1.0
    assert float(np.abs(q[o["mate_nodes"]] - 1.0).max()) == 0.0
    assert float(np.abs(q[o["draw_absorbing_nodes"]]).max()) == 0.0
    assert o["q_residual"] < 1e-9, o["q_residual"]
    # not degenerate: the committor must actually carry a gradient
    interior = q[(q > 0.0) & (q < 1.0)]
    assert interior.size > 1000, interior.size


# ---------------------------------------------------------------------------
# 2. THE EMBEDDING AND THE GRAPH
# ---------------------------------------------------------------------------

def test_the_embedding_is_local_and_leaks_no_label():
    """The encoder may not see the answer. No feature is a function of dtm or
    of the committor -- if it were, the Euclidean control would win by
    construction and the comparison would prove nothing."""
    src = MODULE.read_text(encoding="utf-8")
    body = src[src.index("def embed("):]
    body = body[:body.index("\ndef ")]
    for banned in ("dtm", "q_loss", "committor", "oracle"):
        assert banned not in body, "embed() reads %r" % banned
    X = cs.embed(np.arange(64))
    assert X.ndim == 2 and X.shape[0] == 64
    assert np.isfinite(X).all()


def test_no_import_from_the_untracked_bed_tree():
    src = MODULE.read_text(encoding="utf-8")
    assert "beds" not in src, "the module reaches into the untracked bed tree"


def test_w1_is_inherited_from_curvature_and_not_reimplemented():
    src = MODULE.read_text(encoding="utf-8")
    assert "from ceqjepa.curvature import" in src or "from .curvature import" in src
    for forbidden in ("linprog", "sinkhorn", "def w1"):
        assert forbidden not in src, "%r appears: W1 must come from curvature" % forbidden


def test_pinned_constants_are_stated_in_the_docstring():
    doc = cs.__doc__
    assert "K_NEIGHBORS" in doc or "k     = 8" in doc or "k = 8" in doc
    assert str(K_NEIGHBORS) in doc and str(ALPHA) in doc
    assert cs.EDGE_LENGTH in doc
    assert "SEED" in doc or "seed" in doc
    assert cs.ENDGAME in doc


# ---------------------------------------------------------------------------
# 3. T-STEPS, WITH ITS CONTROLS
# ---------------------------------------------------------------------------

def test_no_geodesic_number_is_reported_without_its_euclidean_control():
    r = cs.steps_report()
    for arm in cs.ARMS:
        assert arm in r, arm
        a = r[arm]
        for field in ("rho", "rho_ci_lo", "rho_ci_hi", "pearson", "n"):
            assert field in a, (arm, field)
    ns = {r[arm]["n"] for arm in cs.ARMS}
    assert len(ns) == 1, ns
    # matched graph: every graph arm is the SAME nodes and the same edge set
    for arm in ("curv", "curv_inv", "hop"):
        assert r[arm]["n_graph_edges"] == r["hop"]["n_graph_edges"]
        assert r[arm]["n_graph_nodes"] == r["n_graph_nodes"]
    assert r["n_graph_nodes"] > r["curv"]["n"]  # the basin anchors are not scored


def test_every_curvature_number_carries_its_edge_count():
    r = cs.steps_report()
    assert r["kappa"]["n_edges"] > 0
    assert r["kappa"]["n_edges"] == r["curv"]["n_graph_edges"]
    assert "refusals" in r and "n_refused" in r
    assert r["k"] == K_NEIGHBORS and r["alpha"] == ALPHA


def test_shuffled_plies_destroy_every_correlation():
    r = cs.steps_report()
    s = r["shuffled"]
    for arm in cs.ARMS:
        assert abs(s[arm]["rho"]) < 0.15, (arm, s[arm]["rho"])
        assert s[arm]["rho_ci_lo"] < 0.0 < s[arm]["rho_ci_hi"], (arm, s[arm])
    # and the unshuffled run must be somewhere else entirely
    assert abs(r["curv"]["rho"]) > 3 * abs(s["curv"]["rho"])


def test_one_basin_only_produces_no_spurious_separation():
    """All positions drawn from a single distance level: the label has no
    variance, so a correlation does not exist. The module must refuse rather than
    return any number at all."""
    one = cs.one_basin_check()
    for arm in cs.ARMS:
        v = one[arm]
        assert is_refusal(v), (arm, v)
        assert "variance" in v.reason.lower() or "constant" in v.reason.lower()


def test_the_geodesic_from_a_position_to_itself_is_exactly_zero():
    r = cs.steps_report()
    assert r["self_geodesic_max"] == 0.0
    assert r["self_euclid_max"] == 0.0


def test_both_length_orientations_are_reported():
    """An affinity is high where two nodes are CLOSE; a length is high where they
    are FAR. Mapping curvature onto a length therefore has a DIRECTION, and the
    direction is load-bearing. Both orientations run as arms so the reader sees
    whether inverting it flips or collapses the result, and the map stays
    strictly positive so no edge is a negative cycle."""
    r = cs.steps_report()
    assert cs.EDGE_LENGTH == "exp(-kappa)"
    assert cs.EDGE_LENGTH_INVERTED == "exp(+kappa)"
    assert r["min_edge_length"] > 0.0
    assert r["min_edge_length_inverted"] > 0.0
    assert r["curv"]["rho"] != r["curv_inv"]["rho"]


def test_the_curvature_arm_is_scored_against_its_control_whichever_way_it_lands():
    """A treatment that loses to its control has to say so in the report, with the
    arithmetic, not only in numbers the reader is left to subtract themselves."""
    r = cs.steps_report()
    assert r["winner"] in cs.ARMS
    assert r["curv_minus_euclid"] == r["curv"]["rho"] - r["euclid"]["rho"]
    assert r["curv_minus_hop"] == r["curv"]["rho"] - r["hop"]["rho"]
    assert r["verdict"] and r["reroute"]


def test_the_null_carries_its_mechanism():
    """Reporting that curvature changed nothing is half a result. The other half
    is what it did instead, and BOTH bounds here are load-bearing.

    Below the upper bound: the two geodesics must NOT be the same ranking. If
    they were, "no effect on the target" would be trivially explained by "the two
    arms were the same arm", and the finding would say nothing about curvature.
    Above the lower bound: the reweighting must still be recognisably a
    reweighting of the same graph rather than a different graph. Between them,
    the only reading left is that curvature reorders and the reordering is
    orthogonal to plies-to-loss -- noise, not a tie.
    """
    r = cs.steps_report()
    assert 0.85 < r["curv_vs_hop_rho"] < 0.999, r["curv_vs_hop_rho"]
    assert r["curv_len_ratio"] > 1.5, r["curv_len_ratio"]
    assert r["mean_path_edges"] > 1.0, r["mean_path_edges"]


def test_the_move_graph_ceiling_is_reported():
    """The exact legal-move graph gives the best any graph geodesic could do.
    Without it the reader cannot tell a weak result from a hard bed."""
    r = cs.steps_report()
    assert "move_graph" in r
    assert r["move_graph"]["n"] == r["curv"]["n"]
    # and it must be reported with the path statistics that stop "the paths were
    # too short" being available as an excuse for the arms that lost
    assert r["mean_move_path_edges"] > 0.0
    assert r["n_move_levels"] > 1 and r["n_hop_levels"] > 1
    assert r["n_label_levels"] > 2


# ---------------------------------------------------------------------------
# 4. THE BUDGET AND THE BANNER
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _demo_run():
    """One execution of the shipped self-check, shared by the tests below."""
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "ceqjepa.chess_steps"],
                       capture_output=True, text=True, timeout=900,
                       cwd=str(MODULE.parent.parent))
    return p, time.time() - t0


@lru_cache(maxsize=1)
def _demo_run_again():
    """A SECOND execution of the shipped self-check.

    Only the variation check needs it, and it costs a full run, but nothing
    cheaper reaches the entry point: a constant can be printed beside a
    measurement inside one run and look exactly like it. Across two runs it
    cannot, because a wall clock does not repeat.
    """
    return subprocess.run([sys.executable, "-m", "ceqjepa.chess_steps"],
                          capture_output=True, text=True, timeout=900,
                          cwd=str(MODULE.parent.parent))


_RATE = re.compile(r"=\s*(\d+\.\d+) ms/edge|^\s*(\d+\.\d+) ms/edge", re.M)


def _published_rates(stdout):
    """Every per-edge rate the run PRINTED, in order."""
    return [g for m in _RATE.finditer(stdout) for g in m.groups() if g]


def test_the_published_price_varies_between_runs():
    """A wall clock does not repeat, and that is the discriminator.

    Every other bind on the price is internal to one run: identities between
    numbers that a literal satisfies as happily as a measurement, because they
    are all downstream of the same root. This one is not. Two timings inside a
    run must differ, and the rate the ENTRY POINT publishes must differ between
    two runs of the entry point. A constant prints the same digits twice and is
    rejected on both counts, without any threshold to tune and without asking
    what machine this is.
    """
    first = _published_rates(_demo_run()[0].stdout)
    second = _published_rates(_demo_run_again().stdout)
    assert len(first) >= 2 and len(second) >= 2, (first, second)
    # two timings inside one run
    assert first[0] != first[1], first
    assert second[0] != second[1], second
    # THE PUBLISHED PRICE ITSELF, element by element -- not the list. Comparing
    # the lists was the first form of this check and an adversarial plant walked
    # through it: a forged constant ball rate was masked by the real, varying
    # probe rate beside it, so the lists differed and the test passed while the
    # published number was frozen. Each position has to vary on its own.
    assert first[0] != second[0], (first[0], second[0])
    assert first[1] != second[1], (first[1], second[1])


def test_self_check_runs_under_300s_and_ends_with_the_banner():
    p, elapsed = _demo_run()
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    assert p.stdout.rstrip().endswith("ALL SELF-CHECKS PASSED"), p.stdout[-2000:]
    assert elapsed < 300.0, "self-check took %.1f s" % elapsed


# Every number this pattern finds must be produced by the run, AS A WHOLE TOKEN.
# NO EXEMPTION LIST. Four shapes, longest-first so each match is maximal: the
# scientific branch consumes all of 9.645e-13 rather than leaving 9.645 to the
# decimal branch, which is what makes token comparison meaningful.
#
# Two defects have been found in this guard, both by planting into a scratch copy
# and watching it pass. (1) SUBSTRING VACUITY: matching with `in` let a planted
# 9.645 be satisfied by the printed 9.645e-13, so a docstring could state the
# leading digits of any longer printed number and be believed. (2) BARE INTEGERS:
# 40,317 was caught and 40317 was not. The second widening is worthless without
# the first -- across the four modules in this round, of 98 bare integers already
# in docstrings, 0 were absent as substrings and 22 were absent as tokens.
_NUMBER = re.compile(r"\d+(?:\.\d+)?[eE][-+]?\d+"      # 9.645e-13, 1e-12
                     r"|\d{1,3}(?:,\d{3})+"              # 368,452
                     r"|\d+\.\d+"                       # 4.133, 0.5
                     r"|\d+")                             # 300, 40317


def _tokens(text):
    """The number tokens in `text`, commas stripped. A set, for exact matching."""
    return {m.group(0).replace(",", "") for m in _NUMBER.finditer(text or "")}


def _every_docstring():
    """Every docstring the MODULE and THIS TEST FILE own, keyed by where it lives.

    Selection is by __module__ rather than by __all__, so demo(), the private
    helpers and the refusal machinery are all in scope. Scanning only the module
    docstring is how the Bonett-Wright constant survived every pass: it sits one
    line below a function docstring, outside the reach of a check that never
    looked there.

    This file is in scope too, and it is the last place a number can hide. A test
    docstring that quotes a figure is making the same claim the module makes, and
    until now no guard in the round scanned its own file.
    """
    out = []
    for mod in (cs, sys.modules[__name__]):
        out.append((mod.__name__ + ":<module>", mod.__doc__))
        for name, obj in sorted(vars(mod).items()):
            if getattr(obj, "__module__", None) != mod.__name__:
                continue
            if getattr(obj, "__doc__", None):
                out.append((mod.__name__ + ":" + name, obj.__doc__))
    return out


def test_the_reroute_price_is_measured_not_asserted():
    """The docstring guard would NOT have caught the strike on this file, and
    saying so is the whole point of this test.

    4.133 ms/edge and 20,250 s were hardcoded into a string demo() printed, so
    they appeared in the output and satisfied "every docstring number is printed
    by a run". What was wrong with them was never that they were absent. It was
    that nothing in the module had computed them -- the module could not even
    build a move-graph curvature. Being printed and being measured are different
    properties, and only the second one is worth anything.

    Pinning the published RATE to seconds/edges is necessary and was not enough.
    move_graph_price is homogeneous of degree one in `seconds`, so replacing that
    one root with a literal carries ms_per_edge, lower_seconds and lower_hours
    along with it and every internal identity still holds. A fabricated clock,
    planted at roughly a third below the measured one, passed this test in full;
    the exact figures are on the board rather than here, because this test is the
    reason a docstring may not quote a number the run cannot produce. The identity
    was bound at the wrong level: it checked the arithmetic downstream of the
    measurement and never the measurement.

    So the root is bound four ways, and they are not equally strong. The load is
    carried by the first: each interval must equal the difference of two RAW
    perf_counter reads that are returned alongside it, with all four reads
    strictly increasing and enclosed by a fifth taken on the way out. That is
    exact and machine-independent, and a literal at the root breaks it at once.
    Then a second clock over different work in the same run, which one literal
    cannot also be. Then the two rates agreeing that the larger, denser ball is
    the slower one per edge, which is a physical fact about the work rather than a
    threshold tuned to an adversary. Then, loosely, a sanity band around the rates
    real machines have produced.

    The band is deliberately loose and it was tight until it kept being wrong. An
    exact span assert fired three times in one afternoon on ONE machine as other
    work finished around it, every time on an honest run. Calibration cannot
    carry discrimination: tightening it converts honest speed into red tests, and
    a fabricated clock only has to land in the range real clocks occupy. So the
    band is a sanity bound and the work is done by the raw-read identity above and
    by test_the_published_price_varies_between_runs, both of which hold on any
    machine at any load.

    The honest limit: someone who fabricates every clock with mutually consistent
    values, AND makes them vary between runs, has built a plausible measurement.
    Nothing here catches that and nothing here pretends to.
    """
    p = cs.move_graph_price()
    assert p["seconds"] > 0.0, p["seconds"]
    assert p["n_attempted"] == p["n_edges"] + p["n_refused"] > 0
    assert abs(p["ms_per_edge"]
               - 1000.0 * p["seconds"] / p["n_attempted"]) < 1e-9
    assert abs(p["lower_seconds"]
               - p["ms_per_edge"] / 1000.0 * p["full_graph_edges"]) < 1e-6
    assert p["full_graph_edges"] > 1_000_000, p["full_graph_edges"]
    assert p["full_degree"] > p["ball_degree"] > 0.0
    assert p["kappa"]["n_edges"] == p["n_edges"]
    # the sentence the report publishes must carry the value that was measured
    assert ("%.3f" % p["ms_per_edge"]) in cs.steps_report()["reroute"]

    # THE ROOT, exactly: each interval is the difference of two raw reads that
    # come back with it, and the reads are strictly ordered and enclosed.
    assert p["t_probe0"] < p["t_probe1"] < p["t_ball0"] < p["t_ball1"] < p["t_return"]
    assert p["seconds"] == p["t_ball1"] - p["t_ball0"]
    assert p["probe_seconds"] == p["t_probe1"] - p["t_probe0"]
    assert p["t_return"] - p["t_probe0"] >= p["seconds"] + p["probe_seconds"]

    # THE ROOT, corroborated. Two wall clocks in one run, over different work.
    assert p["probe_seconds"] > 0.0, p["probe_seconds"]
    assert p["probe_attempted"] > 0
    assert p["seconds"] != p["probe_seconds"], "two timings returned one value"
    assert abs(p["probe_ms_per_edge"]
               - 1000.0 * p["probe_seconds"] / p["probe_attempted"]) < 1e-9
    # the bigger, denser ball is the slower one per edge, every time
    assert 1.0 < p["ms_per_edge"] / p["probe_ms_per_edge"] < 2.0, (
        "the two clocks disagree by %.3fx about what machine this is"
        % p["rate_ratio"])

    # AND a LOOSE sanity band around the rates real machines have produced. Not
    # the discriminator -- see the docstring -- just a bound on the absurd.
    lo, hi = cs.PRICE_OBSERVED_MS_PER_EDGE
    assert lo < hi
    assert p["rate_in_band"] is True, (
        "this run measured %.3f ms/edge, more than %gx outside the recorded span "
        "(%.3f, %.3f). If the run is honest, widen PRICE_OBSERVED_MS_PER_EDGE to "
        "record the new observation."
        % (p["ms_per_edge"], cs.PRICE_BAND_FACTOR, lo, hi))


def test_the_move_graph_has_no_reciprocal_arcs():
    """The reason W.nnz // 2 equals the directed arc count, made executable.

    The move relation is bipartite by side to move: every arc runs between a
    White-to-move position and a Black-to-move one, and an arc's reverse would
    have to be a move by the side that did not just move. So no arc has a
    reciprocal and symmetrising doubles the nonzeros exactly. That was true when
    it was written and it was bound to nothing -- correct arithmetic resting on
    prose. Counted here directly from the arc list, and required to be zero.
    """
    sp = cs.space()
    n = sp["n_states"]
    key = np.sort(sp["pred"].astype(np.int64) * n + sp["succ"])
    rev = sp["succ"].astype(np.int64) * n + sp["pred"]
    idx = np.clip(np.searchsorted(key, rev), 0, key.size - 1)
    assert int((key[idx] == rev).sum()) == 0, "the move graph has reciprocal arcs"
    # and the module must REPORT the count, not leave it to this test
    p = cs.move_graph_price()
    assert p["n_reciprocal_arcs"] == 0
    assert p["full_graph_edges"] == sp["n_edges"]


def test_the_retired_price_is_withdrawn_as_unreproducible_not_as_wrong():
    """A wall clock cannot convict another wall clock.

    An earlier revision of this repair called the retired 4.133 ms/edge "1.9x
    wrong" on the strength of one timing on one loaded box. Running the module's
    own move_graph_price on a second box returned 4.116 ms/edge, within 0.4% of
    it: the retired figure reproduces. It was struck for being UNREPRODUCIBLE --
    nothing in the module measured it -- and that is the only charge that stands.
    The price is therefore a spread across machines and loads, and the published
    sentence has to carry both ends of it rather than whichever end this box
    happened to produce.
    """
    doc = cs.__doc__
    assert "1.9x" not in doc, "the withdrawn overreach is still on the page"
    assert "UNREPRODUCIBLE" in doc or "unreproducible" in doc.lower()
    lo, hi = cs.PRICE_OBSERVED_MS_PER_EDGE
    assert 0.0 < lo < hi
    reroute = cs.steps_report()["reroute"]
    assert ("%.3f" % lo) in reroute and ("%.3f" % hi) in reroute, reroute
    assert "lower bound on a lower bound" in reroute.lower()


def test_the_module_can_actually_build_the_object_it_prices():
    """The structural check the Inspector used, kept as a test. A price for
    curvature on a move graph is not credible from a module that imports no
    graph-curvature entry point -- that absence is what proved the figures came
    from somewhere else."""
    src = MODULE.read_text(encoding="utf-8")
    assert "curvature_from_graph" in src
    p = cs.move_graph_price()
    assert p["n_edges"] > 100, p["n_edges"]
    assert np.isfinite(p["kappa"]["mean"])


def test_every_number_in_every_docstring_is_printed_by_the_demo():
    """A number the shipped artefact states as fact must be produced by the
    shipped artefact, in the run any reader can repeat.

    This is the guard that caught the strike on this file. The reroute price --
    a per-edge rate, a ball edge count, kappa statistics for a graph this module
    could not build -- was real when measured, but it was measured in a scratch
    script that never entered the tree, and then printed by demo() as though the
    module had produced it. Nothing in the module or the tests could have caught
    that, because consistent arithmetic on an unmeasured rate looks exactly like
    consistent arithmetic on a measured one.

    Matching is TOKEN against TOKEN, with commas stripped on both sides. Substring
    matching was the earlier form and it was vacuous: a planted bare decimal was
    satisfied by any longer printed number that began with those digits, so a
    docstring could quote the head of a real figure and be believed. A docstring
    number must now equal a number the run actually printed.
    """
    p, _ = _demo_run()
    assert p.returncode == 0, p.stderr[-4000:]
    printed = _tokens(p.stdout)
    docs = _every_docstring()
    scanned = {n for _, doc in docs for n in _tokens(doc)}

    # A GUARD WHOSE EVIDENCE IS MISSING REPORTS WHAT A GUARD THAT PASSED REPORTS.
    # "0 missing" is reachable by scanning no docstrings, or by comparing against
    # an output that never arrived, and neither would look any different below.
    # Both sides are therefore required to be substantial before the comparison
    # is allowed to mean anything.
    assert p.stdout.rstrip().endswith("ALL SELF-CHECKS PASSED"), p.stdout[-2000:]
    assert len(docs) >= 30, "only %d docstrings in scope" % len(docs)
    assert len(scanned) >= 60, "only %d numbers scanned" % len(scanned)
    assert len(printed) >= 60, "only %d numbers captured from the run" % len(printed)

    missing = sorted({(where, n) for where, doc in docs
                      for n in _tokens(doc) if n not in printed})
    assert not missing, ("%d numbers across %d docstrings are printed by no run: %r"
                         % (len(missing), len(docs), missing))
