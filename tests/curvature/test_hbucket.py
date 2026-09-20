"""Tests for ceqjepa.hbucket: does bucketing by k-ply isomorphism leave any
distance-to-mate variance behind that is not already a board symmetry?

WRITTEN RED. Every test here existed and failed before ceqjepa/hbucket.py did.
The verbatim first failure is recorded in the module docstring of hbucket.py.

The planted negatives, which must be SEEN to fire and are the reason this file
is not a rubber stamp:

  (a) a graph on which every rooted neighbourhood IS isomorphic must come back
      as ONE bucket. A bucketing that splits it is reading node identity, not
      structure -- which is exactly the bug the first probe of this round had,
      where the arc-direction label was written as 'forward' when the tail's
      integer index was the smaller one. That label is not isomorphism
      invariant, and it silently split every symmetry orbit into singletons.
  (b) a deliberately non-isomorphic pair must NOT share a bucket.
  (c) on a bed where the label is a pure function of out-degree, the
      within-bucket label variance must be exactly 0.0. That is the design's
      own null: if the design reads content there it is reading noise.

The guard standard is the five-clause one plus the identity layer, copied in
form from tests/curvature/test_lifetimes.py:535 and its vacuity controls at
:615, not in content -- no number is carried across from another agent's run.
"""

import ast
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _hb():
    from ceqjepa import hbucket
    return hbucket


def _cv():
    from ceqjepa import curvature
    return curvature


# ---------------------------------------------------------------------------
# 1. THE PLANTED NEGATIVES
# ---------------------------------------------------------------------------

def test_a_bed_of_isomorphic_neighbourhoods_lands_in_exactly_one_bucket():
    """C_20 is vertex- and edge-transitive: every rooted edge ball is the same.

    This is the test the first probe of this round would have failed. It fires
    on any bucket key that is a function of node identity rather than of
    structure, and it fires at every k, because a cycle is homogeneous at every
    radius short of wrapping.
    """
    m = _hb()
    G = nx.cycle_graph(20)
    adj = m.Adj.from_networkx(G)
    edges = [(u, v) for u, v in G.edges()]
    assert len(edges) == 20, "the bed changed shape"
    for k in (1, 2, 3):
        buckets = m.bucket(adj, edges, k)
        assert len(buckets) == 1, (
            "C_20 at k=%d split into %d buckets: the key is reading node "
            "identity, not structure" % (k, len(buckets)))
        only = next(iter(buckets.values()))
        assert len(only) == 20, "the one bucket lost edges: %d of 20" % len(only)


def test_a_deliberately_non_isomorphic_pair_never_shares_a_bucket():
    """A triangle edge and a path edge, at k=1, are different rooted balls.

    Kept deliberately small so an exact isomorphism check settles it in the
    same test: WL agreeing is not evidence unless the exact check agrees too.
    """
    m = _hb()
    tri = nx.complete_graph(3)
    pth = nx.path_graph(3)
    a = m.Adj.from_networkx(tri)
    b = m.Adj.from_networkx(pth)
    Ga = m.rooted_neighbourhood(a, 0, 1, 1)
    Gb = m.rooted_neighbourhood(b, 0, 1, 1)
    assert not m.exact_isomorphic(Ga, Gb), \
        "a triangle and a path were called isomorphic by the exact check"
    assert m.wl_key(Ga) != m.wl_key(Gb), \
        "WL gave a triangle edge and a path edge the same bucket key"


def test_the_designs_own_null_reads_exactly_empty():
    """Label a pure function of out-degree: within-bucket variance must be 0.0.

    NOT "small". A bucket is an isomorphism class, degree is an isomorphism
    invariant, and a label that is a function of degree is therefore constant
    on a bucket by theorem. Any nonzero number here means the buckets are not
    isomorphism classes and every downstream reading is void.

    The vacuity control is the non-singleton count: a bed that produced only
    singletons would pass a variance test for the reason that there is nothing
    in it, so the count is asserted before the variance is.
    """
    m = _hb()
    G = m.degree_determined_bed()
    adj = m.Adj.from_networkx(G)
    label = {u: float(G.degree(u)) for u in G}
    edges = [(u, v) for u, v in G.edges()]
    buckets = m.bucket(adj, edges, 1)
    sizes = [len(v) for v in buckets.values()]
    n_multi = sum(1 for s in sizes if s > 1)
    assert n_multi >= 2, (
        "VACUOUS BED: only %d non-singleton buckets, so a zero variance below "
        "would be measuring an empty set" % n_multi)
    worst = max(
        (max(label[x] for x, _ in v) - min(label[x] for x, _ in v))
        for v in buckets.values())
    assert worst == 0.0, (
        "a label that is a pure function of degree varied by %r inside a "
        "bucket: the buckets are not isomorphism classes" % worst)


# ---------------------------------------------------------------------------
# 2. THE SYMMETRY THAT DECIDES THE DESIGN
# ---------------------------------------------------------------------------

def test_d4_is_an_exact_automorphism_and_it_carries_dtm_and_the_committor():
    """All eight board symmetries map the space to itself and fix the labels.

    If this holds then the buckets the design produces contain symmetry orbits,
    and distance-to-mate is CONSTANT on a symmetry orbit -- so the variance the
    design proposes to read as consequence is zero there by the same theorem
    that makes the local statistics constant. The whole pre-kill turns on it,
    so it is measured over every transform rather than asserted from the fact
    that chess has no pawns here.
    """
    m = _hb()
    rep = m.d4_report()
    assert len(rep["per_transform"]) == 8, \
        "D4 has eight elements, the report has %d" % len(rep["per_transform"])
    assert rep["n_positions_checked"] >= 2000, \
        "only %d positions checked: too few to bind" % rep["n_positions_checked"]
    for name, row in rep["per_transform"].items():
        assert row["missing"] == 0, \
            "%s mapped %d positions outside the space" % (name, row["missing"])
        assert row["dtm_mismatch"] == 0, \
            "%s changed distance-to-mate on %d positions" % (name, row["dtm_mismatch"])
        assert row["q_mismatch"] == 0, \
            "%s changed the committor on %d positions" % (name, row["q_mismatch"])
    assert rep["total_checks"] == 8 * rep["n_positions_checked"]


def test_the_bucketing_puts_a_real_symmetry_orbit_in_one_bucket():
    """The real-data form of planted negative (a), and the harder one.

    An orbit of the board symmetry group is a set of edges whose rooted
    neighbourhoods are isomorphic BY CONSTRUCTION. If the bucket key does not
    put them together, the key is broken, and the census that follows it counts
    nothing. The first probe of this round failed exactly here.
    """
    m = _hb()
    g = m.graph()
    orbits = m.sample_orbits(g, n_orbits=12, seed=m.SEED)
    full = [o for o in orbits if len(o) == 8]
    assert len(full) >= 8, \
        "only %d of %d sampled orbits had all eight members" % (len(full), len(orbits))
    for k in (1, 2):
        for orb in full[:6]:
            keys = {m.wl_key(m.rooted_neighbourhood(g.adj, x, y, k)) for x, y in orb}
            assert len(keys) == 1, (
                "at k=%d a symmetry orbit of %d edges split into %d buckets: "
                "the key is not isomorphism invariant" % (k, len(orb), len(keys)))


def test_wl_collisions_are_measured_with_an_exact_check_not_assumed_away():
    """A collision rate you did not measure is a theorem you do not have.

    weisfeiler_lehman_graph_hash is not a complete isomorphism invariant, so
    every bucket that merges two distinct symmetry orbits is re-checked with
    networkx.is_isomorphic, roots pinned by a categorical node match, under a
    deterministic step budget.

    THREE OUTCOMES, NOT TWO. The matcher does not always terminate on these
    balls, so a bucket it could not decide is counted as undecided rather than
    folded into either answer. Folding an exhausted search into "collision"
    would fabricate the very number the audit exists to measure.
    """
    m = _hb()
    aud = m.collision_audit(k=1, n_orbits=m.AUDIT_ORBITS, seed=m.SEED)
    assert aud["n_edges"] > 0 and aud["n_orbits"] <= m.AUDIT_ORBITS
    assert aud["n_cross_orbit_buckets"] == (
        aud["n_real_isomorphism"] + aud["n_wl_collision"] + aud["n_undecided"]), \
        "the audit lost a bucket between the three outcomes"
    assert aud["n_decided"] == aud["n_real_isomorphism"] + aud["n_wl_collision"]
    assert 0.0 <= aud["collision_rate"] <= 1.0
    if aud["n_decided"]:
        assert aud["collision_rate"] == aud["n_wl_collision"] / aud["n_decided"]
    else:
        assert aud["collision_rate"] == 0.0


def test_the_exact_matcher_reports_running_out_rather_than_guessing():
    """A bounded search that returns False on exhaustion is a fabrication.

    Given a budget of one candidate pair the matcher cannot decide anything, so
    it must return None. If it returns False here it is returning False for
    every hard pair in the audit too, and the collision rate is invented.
    """
    m = _hb()
    G = nx.complete_graph(6)
    H = nx.complete_graph(6)
    for K in (G, H):
        nx.set_node_attributes(K, {u: "." for u in K}, "r")
    assert m.exact_isomorphic(G, H, budget=1) is None, \
        "the matcher decided a question it had no budget to search"
    assert m.exact_isomorphic(G, H) is True
    assert m.exact_isomorphic(G, nx.path_graph(6)) is False


def test_the_k_local_statistics_are_exactly_constant_inside_a_bucket():
    """The empirical consequence of "isomorphic, therefore constant".

    Out-degree, ln out-degree and mean successor out-degree are k-local at
    k >= 1, so an isomorphism class fixes all three. The spread must read 0.0
    exactly. It is checked on the real graph, where a broken key is what would
    make it nonzero, and the non-singleton count is asserted first so a pass
    cannot come from an empty census.
    """
    m = _hb()
    aud = m.constancy_audit(k=1, n_orbits=m.AUDIT_ORBITS, seed=m.SEED)
    assert aud["n_nonsingleton"] >= 10, \
        "only %d non-singleton buckets: the spread below is vacuous" % aud["n_nonsingleton"]
    for stat in ("out_degree", "ln_out_degree", "successor_out_degree"):
        assert aud["spread"][stat] == 0.0, (
            "%s spread %r inside a bucket: the buckets are not isomorphism "
            "classes and stage two would be meaningless" % (stat, aud["spread"][stat]))


# ---------------------------------------------------------------------------
# 3. THE CENSUS IS A SAMPLE AND SAYS SO
# ---------------------------------------------------------------------------

def test_the_census_reports_its_sample_its_seed_and_its_population():
    m = _hb()
    rows = m.census_table()
    assert [r["k"] for r in rows] == list(m.K_PLIES)
    for r in rows:
        assert r["is_sample"] is True, "the census must say it is a sample"
        assert r["seed"] == m.SEED
        assert r["n_edges_sampled"] < r["population_edges"], \
            "a sample that is the whole population is not a sample"
        assert r["n_buckets"] >= 1
        assert r["n_nonsingleton"] <= r["n_buckets"]
        assert r["n_nonzero_dtm_var"] <= r["n_nonsingleton"]
        assert r["largest_bucket"] >= 1
        assert r["n_edges_sampled"] == sum(r["bucket_sizes"].values())


def test_dtm_variance_inside_a_bucket_is_computed_over_both_endpoints():
    """A variance read on the tail only would miss a bucket that varies at the head."""
    m = _hb()
    g = m.graph()
    orbit = m.sample_orbits(g, n_orbits=1, seed=m.SEED)[0]
    v = m.bucket_dtm_variance(g, orbit)
    assert set(v) == {"tail", "head", "worst"}
    assert v["worst"] == max(v["tail"], v["head"])


# ---------------------------------------------------------------------------
# 4. THE H-TRANSFORM CURVATURE
# ---------------------------------------------------------------------------

def test_the_h_kernel_is_a_kernel_and_the_h_transform_is_not_the_bare_walk():
    """Rows of p_h sum to 1 where h > 0, and p_h differs from p where h varies."""
    m = _hb()
    g = m.graph()
    rng = np.random.default_rng(m.SEED)
    idx = rng.choice(g.n_live_edges, 200, replace=False)
    differ = 0
    for e in idx:
        x = int(g.src[e])
        supp, mass = m.h_kernel_row(g, x)
        assert abs(float(mass.sum()) - 1.0) < 1e-12, \
            "p_h row at %d sums to %r" % (x, float(mass.sum()))
        assert (mass >= 0.0).all()
        bare = np.full(supp.size, 1.0 / supp.size)
        if np.abs(mass - bare).max() > 1e-9:
            differ += 1
    assert differ >= 150, (
        "only %d of 200 rows differ from the uniform walk: the h-transform is "
        "doing nothing and kappa_h is kappa" % differ)


def test_kappa_h_reduces_to_the_shipped_instrument_when_h_is_constant():
    """CROSS-INSTRUMENT IDENTITY, and the only reason to trust the new code.

    With h constant the Doob transform is the identity on the kernel, so
    kappa_h on a symmetric unweighted graph must equal ceqjepa.curvature's
    kappa_edge term for term. Compared POSITION BY POSITION, never with one
    allclose over a list, because one frozen entry hides beside neighbours
    that still move.
    """
    m, cv = _hb(), _cv()
    G = nx.les_miserables_graph()
    G = nx.convert_node_labels_to_integers(G)
    W = nx.to_numpy_array(G, weight=None)
    D = cv.graph_metric(W)
    adj = m.Adj.from_networkx(G)
    h = np.ones(W.shape[0])
    edges = sorted(G.edges())[:40]
    assert len(edges) == 40
    for pos, (i, j) in enumerate(edges):
        mine = m.kappa_h_edge(adj, h, D, i, j, alpha=cv.ALPHA)
        theirs = cv.kappa_edge(W, D, i, j, alpha=cv.ALPHA)
        assert not cv.is_refusal(mine) and not cv.is_refusal(theirs), \
            "edge %d (%d,%d) refused: %r %r" % (pos, i, j, mine, theirs)
        assert abs(float(mine) - float(theirs)) < 1e-9, (
            "position %d, edge (%d,%d): kappa_h %r != kappa %r with h constant"
            % (pos, i, j, mine, theirs))


def test_the_committor_hat_check_partials_out_the_four_cheap_moments():
    """kappa_h regressed on h(x), h(y), Var_N(x) h, Var_N(y) h.

    The design is only alive if kappa_h carries something those four do not.
    The R^2 is reported with the count it came from and a CI; a check that
    reported a number with no count is the one this project keeps repeating.
    """
    m = _hb()
    res = m.committor_hat_check()
    assert res["n"] >= 200, "only %d edges: the R^2 has no count" % res["n"]
    assert res["design_columns"] == ["h_x", "h_y", "var_h_nx", "var_h_ny"]
    assert 0.0 <= res["r2"] <= 1.0 + 1e-12
    lo, hi = res["ci"]
    assert lo <= res["r2"] <= hi
    assert res["residual_sd"] >= 0.0


def test_the_arrow_is_checked_against_the_ground_truth_it_has():
    """Kill 3. DTM strictly decreases along optimal play, so the arrow is known.

    The asymmetry of kappa_h is measured on the symmetrised move+unmove graph
    and its sign is compared with the direction distance-to-mate actually
    falls. An asymmetry that does not agree with the arrow it claims to have
    found is not a reading of the arrow.
    """
    m = _hb()
    res = m.arrow_check()
    assert res["n"] >= 100, "only %d edges" % res["n"]
    assert 0.0 <= res["agree_rate"] <= 1.0
    assert res["agree_rate"] == res["n_agree"] / res["n"]
    assert res["n_agree"] + res["n_disagree"] + res["n_tied"] == res["n"]
    assert res["median_abs_asymmetry"] >= 0.0


# ---------------------------------------------------------------------------
# 5. THE BUDGET
# ---------------------------------------------------------------------------

def test_the_module_runs_under_its_stated_bar_and_prints_the_pass_line():
    """python -m ceqjepa.hbucket, under 300 s, ending on the exact line."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ceqjepa.hbucket"],
            capture_output=True, text=True, timeout=300,
            cwd=str(Path(__file__).resolve().parents[2]))
    except subprocess.TimeoutExpired:
        # The 300 s bar was MEASURED and not ENFORCED: `subprocess.run` carried
        # no `timeout=`, so the assert at the end of this function could only
        # report an overrun after the child had already finished. One run of
        # this test was observed still going at ~12.5 min, i.e. 2.5x the bar it
        # declares, and nothing in the suite could stop it -- which is how a
        # suite becomes unbounded while every test in it claims a budget.
        raise AssertionError(
            "python -m ceqjepa.hbucket did not finish inside the 300 s bar "
            "this test declares. The bar is now enforced by the child's own "
            "timeout, so an over-budget run is a RED at 300 s instead of an "
            "unbounded wait.")
    elapsed = time.time() - t0
    assert proc.returncode == 0, \
        "exit %d\n%s" % (proc.returncode, proc.stderr[-3000:])
    assert proc.stdout.rstrip().endswith("ALL SELF-CHECKS PASSED"), \
        "last line was %r" % proc.stdout.rstrip().splitlines()[-1:]
    assert elapsed < 300.0, "the module took %.1f s against a 300 s bar" % elapsed
    print("\n  module run %.1f s against the 300 s bar" % elapsed)


# ---------------------------------------------------------------------------
# 6. THE GUARD
# ---------------------------------------------------------------------------

#: A number WITH ITS BOUNDARIES. Token matching, not substring: "3.7" is a
#: substring of "13.72" and a substring test would pass a stale number that
#: merely shares digits with a printed one. The inner (?:\.\d+)* keeps a dotted
#: version string as ONE token. The trailing guard is (?!\w)(?!\.\d) and not the
#: tighter (?![\w.]), because a sentence-final period must not hide the number
#: in front of it -- a guard that misses the run's own numbers is the same
#: defect pointing the other way. Integers match as well as decimals, and only
#: after the boundaries are in place.
_MEASURED = re.compile(
    r"(?<![\w.])[+-]?\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")

_CHILD = "HBUCKET_GUARD_CHILD"


def _tokens(text):
    """Every number in `text` as a boundary-anchored token, leading + stripped."""
    return {t.lstrip("+") for t in _MEASURED.findall(text)}


def _own_test_run_output():
    """stdout of THIS file's own tests, so numbers in THEIR docstrings are bound."""
    env = dict(os.environ)
    env[_CHILD] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(Path(__file__).resolve()),
         "-q", "-s", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env,
        cwd=str(Path(__file__).resolve().parents[2]))
    assert proc.stdout, "the child run printed nothing: %r" % (proc.stderr[-400:],)
    return proc.stdout


def _module_docstrings(mod):
    """(name, docstring) for the module and everything DEFINED in it.

    Scoped by __module__ so numpy's and networkx's docstrings do not come
    along. Not scoped by __all__, because demo() is the function that prints
    every number and an __all__ scan is free to miss it. This is a scope rule,
    not an exemption list: no NUMBER is ever exempted anywhere in this file.
    """
    docs = [("module", mod.__doc__ or "")]
    for name, obj in sorted(vars(mod).items()):
        if getattr(obj, "__module__", None) != mod.__name__:
            continue
        doc = getattr(obj, "__doc__", None)
        if isinstance(doc, str) and doc.strip():
            docs.append((name, doc))
        for attr in sorted(vars(obj)) if isinstance(obj, type) else ():
            member = getattr(obj, attr, None)
            member = member.fget if isinstance(member, property) else member
            if getattr(member, "__module__", None) != mod.__name__:
                continue
            sub = getattr(member, "__doc__", None)
            if isinstance(sub, str) and sub.strip() and attr != "__doc__":
                docs.append(("%s.%s" % (name, attr), sub))
    return docs


def _counter_by_ast(paths):
    """SECOND, INDEPENDENTLY WRITTEN COUNTER over the same numbers.

    The first counter walks the imported objects and reads __doc__. This one
    never imports anything: it parses the source with ast and takes the
    docstring of every module, class and function node. Two counters that
    share no code, so a coverage figure that drifts in one is caught by the
    other rather than by nobody.
    """
    total = 0
    for path in paths:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    total += len(_MEASURED.findall(doc))
    return total


def test_every_measured_number_in_a_docstring_is_printed_by_a_run(capsys):
    """No exemption list. A number no run prints cannot be checked by anyone."""
    if os.environ.get(_CHILD):
        pytest.skip("child of the own-file scan; the parent does the scanning")
    m = _hb()
    m.demo()
    out = capsys.readouterr().out
    run = _tokens(out) | _tokens(_own_test_run_output())

    # VACUITY CONTROLS. Two numbers that no run here produces, and one that
    # every run does, so a regex that stopped biting fails instead of passing.
    assert "20260913" in run, "the run stopped printing the seed"
    assert "0913" not in run, \
        "SUBSTRING VACUITY: a fragment of 20260913 is being read as a number"
    assert "41.77" not in run, "the absent-decimal control was found in the run"
    assert "918273" not in run, "the absent-integer control was found in the run"

    docs = _module_docstrings(m) + _module_docstrings(sys.modules[__name__])
    missing = [(where, tok) for where, doc in docs
               for tok in _tokens(doc) if tok not in run]
    checked = sum(len(_tokens(doc)) for _, doc in docs)
    by_ast = _counter_by_ast([Path(m.__file__), Path(__file__)])
    print("\n  %d distinct numbers across %d docstrings, %d missing; "
          "the ast counter saw %d occurrences" % (checked, len(docs), len(missing), by_ast))
    assert checked >= 40, "only %d numbers found: the regex is not biting" % checked
    assert by_ast >= checked, \
        "the ast counter saw %d occurrences under %d distinct tokens" % (by_ast, checked)
    assert not missing, ("these docstring numbers are printed by no run:\n    "
                         + "\n    ".join("%s: %s" % (w, t) for w, t in missing))


def test_the_published_numbers_are_recomputed_here_not_copied(capsys):
    """PRINTED IS NOT MEASURED, and PRINTED BY ANOTHER AGENT IS NOT MEASURED AT ALL.

    Every figure demo() publishes is pinned to an expression that recomputes it
    from the module's own functions in this same process. Position by position,
    never one allclose over a list.
    """
    m = _hb()
    stats = m.demo()
    out = capsys.readouterr().out
    assert isinstance(stats, dict) and stats, "demo() must return what it measured"

    rows = m.census_table()
    assert len(rows) == len(m.K_PLIES) == len(stats["census"])
    for pos, (got, want) in enumerate(zip(stats["census"], rows)):
        for field in ("k", "n_edges_sampled", "n_buckets", "n_nonsingleton",
                      "n_nonzero_dtm_var", "largest_bucket", "population_edges"):
            assert got[field] == want[field], (
                "census row %d field %s: demo said %r, the recomputation says %r"
                % (pos, field, got[field], want[field]))

    d4 = m.d4_report()
    assert stats["d4_total_checks"] == d4["total_checks"]
    assert stats["d4_mismatches"] == sum(
        r["dtm_mismatch"] + r["q_mismatch"] + r["missing"]
        for r in d4["per_transform"].values())

    aud = m.collision_audit(k=1, n_orbits=m.AUDIT_ORBITS, seed=m.SEED)
    assert stats["collision_rate"] == aud["collision_rate"]
    assert stats["collision_n"] == aud["n_cross_orbit_buckets"]
    assert stats["collision_real"] == aud["n_real_isomorphism"]

    con = m.constancy_audit(k=1, n_orbits=m.AUDIT_ORBITS, seed=m.SEED)
    for stat in ("out_degree", "ln_out_degree", "successor_out_degree"):
        assert stats["spread"][stat] == con["spread"][stat], \
            "spread[%s] is a literal, not a measurement" % stat

    hat = m.committor_hat_check()
    assert stats["hat_r2"] == hat["r2"] and stats["hat_n"] == hat["n"]
    arrow = m.arrow_check()
    assert stats["arrow_agree_rate"] == arrow["agree_rate"]
    assert stats["arrow_n"] == arrow["n"]

    for key in ("d4_total_checks", "collision_n", "hat_n", "arrow_n"):
        assert str(stats[key]) in out, \
            "%s = %r is not in the run's output" % (key, stats[key])
    assert ("%.4f" % stats["hat_r2"]) in out or ("%.6f" % stats["hat_r2"]) in out


def test_the_verdict_line_is_derived_from_the_census_not_typed(capsys):
    """The pre-kill verdict must be a function of the numbers, not a string."""
    m = _hb()
    stats = m.demo()
    capsys.readouterr()
    total_live = sum(r["n_nonzero_dtm_var"] for r in stats["census"])
    assert stats["prekill_empty"] == (total_live == 0)
    assert stats["prekill_empty_at_k3"] == (
        [r for r in stats["census"] if r["k"] == 3][0]["n_nonzero_dtm_var"] == 0)
