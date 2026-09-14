"""The ceiling of the unseen-goal committor measure, bound before any model exists.

RED FIRST. Every assertion here was written and run before `ceqjepa/goal_family.py`
existed. Verbatim, `python -m pytest tests/curvature/test_goal_family.py -x -q`
at cf9b0d2 on WIN-16QAL06O9GB:

    ImportError while importing test module
    'C:\\Users\\seal\\Desktop\\New folder (32)\\tests\\curvature\\test_goal_family.py'.
    ...
    E   ImportError: cannot import name 'goal_family' from 'ceqjepa'
    ...
    1 error in 8.83s

WHAT THESE BIND is the part of the measure that can be wrong SILENTLY, which is
the failure mode that killed DCM-1 (`ceqjepa/hf/README.md`: a bed-oracle
headroom of +0.0935 with the wrong sign, found only after training):

  - the D4 quotient is a real lumping. A quotient that is not strongly lumpable
    puts every committor below a DIFFERENT chain's committor, with no symptom
    anywhere downstream;
  - the labels agree with a SIMULATION, which shares none of the boundary-
    condition code that the fixpoint and the LU both route through. This is the
    only check here that an absorbing set wired up backwards cannot pass;
  - the truncated resolvents are nested under the exact committor
    (q1 <= q2 <= q), which an off-by-one or sign-flipped resolvent violates;
  - the pooled shallow fit does not clear KILL_R2;
  - `demo()` exits 0 (L-SURFACE: a check that cannot run is red, never skipped).
"""
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

from ceqjepa import goal_family as gf

#: `python -m ceqjepa.goal_family --draws 8 --train 4`, cf9b0d2,
#: WIN-16QAL06O9GB (python 3.11.9, numpy 2.4.6, scipy 1.17.1).
SMALL = dict(r2_linear=0.8989, r2_linear_within=0.7067, headroom=0.8201,
             marginal_pooled=0.6554)


@pytest.fixture(scope="module")
def quo():
    return gf.quotient()


@pytest.fixture(scope="module")
def small():
    return gf.report(n_test=8, n_train=4, seed=gf.SEED)


def test_orbit_count_is_rederived_not_quoted(quo):
    """46,137 orbits, and DTM constant on every one of them."""
    assert quo["n_positions"] == 368452
    assert quo["n_moves"] == 4891672
    assert quo["n_orbits"] == 46137
    dtm_full, orb = quo["dtm_full"], quo["orbit_of"]
    hi = np.full(quo["n_orbits"], -10 ** 9)
    lo = np.full(quo["n_orbits"], 10 ** 9)
    np.maximum.at(hi, orb, dtm_full)
    np.minimum.at(lo, orb, dtm_full)
    assert int((hi != lo).sum()) == 0


def test_quotient_is_strongly_lumpable(quo):
    """Every one of the 368,452 positions has its orbit's row. Exhaustive."""
    assert quo["lumpability_disagreements"] == 0
    rows = np.asarray(quo["P"].sum(axis=1)).ravel()
    assert float(np.abs(rows[~quo["absorbing"]] - 1.0).max()) < 1e-12
    assert float(np.abs(rows[quo["absorbing"]]).max()) == 0.0


def test_the_label_agrees_with_a_simulation_that_shares_no_code(quo):
    """An INDEPENDENT algorithm for q_{B',C}: walk the chain and see where it
    lands first. The fixpoint and the LU share this module's boundary-condition
    code; a simulated walk reads only P's rows, so a set made absorbing the
    wrong way round cannot survive this and can survive the other two."""
    P, N = quo["P"], quo["n_states"]
    rng = np.random.default_rng(gf.SEED)
    B, C, interior = gf.draw_goal(rng, quo["absorbing"])
    q, res, _ = gf.committor(P, B, C, interior)
    assert res < 1e-9

    indptr, indices, data = P.indptr, P.indices, P.data
    flat = np.cumsum(data)
    head = np.minimum(indptr[:-1], data.size - 1)
    counts = np.diff(indptr)
    cum = flat - np.repeat(flat[head] - data[head], counts)
    gcum = cum + np.repeat(np.arange(N), counts)      # monotone across rows

    walkers, probes = 800, 200
    stop = B | C | quo["absorbing"]
    probe = rng.choice(np.flatnonzero(interior), size=probes, replace=False)
    state = np.repeat(probe, walkers)
    alive = np.ones(state.size, bool)
    hit = np.zeros(state.size, bool)
    for _ in range(4000):
        a = np.flatnonzero(alive)
        if a.size == 0:
            break
        k = np.searchsorted(gcum, state[a] + rng.random(a.size), side="left")
        nxt = indices[np.minimum(k, indices.size - 1)]
        state[a] = nxt
        done = stop[nxt]
        hit[a[done]] = B[nxt[done]]
        alive[a[done]] = False
    assert not alive.any(), f"{int(alive.sum())} walkers never absorbed"

    emp = hit.reshape(probes, walkers).mean(1)
    exact = q[probe]
    # 1/sqrt(n) noise, no bias: measured mean|emp-exact| 0.0043 / 0.0020 / 0.0010
    # at 800 / 3200 / 12800 walkers (scratchpad p3_montecarlo.py, cf9b0d2).
    assert float(np.abs(emp - exact).mean()) < 0.010
    assert float(np.abs(emp - exact).max()) < 0.08


def test_the_checks_above_can_actually_fail(quo):
    """V-3: a check that cannot fail is not a check. Three deliberate
    corruptions, each of which the checks above must catch. Measured, verbatim
    from scratchpad/p7_positive_controls.py at cf9b0d2 on WIN-16QAL06O9GB:
    31 disagreements from moving ONE position to a neighbouring orbit,
    8 from splitting the largest orbit (which has 8 members), max(q2-q) = 1.0000
    from failing to kill C in the 2-hop term, and mean|wrong-right| = 0.8288
    from solving the committor toward C instead of B'.
    """
    from ceqjepa.chess_steps import space

    sp = space()
    n, sink, nt = sp["n_positions"], sp["sink"], sp["n_states"]
    orbit_of, P = quo["orbit_of"], quo["P"]
    orb_all = np.empty(nt, np.int64)
    orb_all[:n] = orbit_of
    orb_all[sink] = quo["n_orbits"]
    src, dst = sp["pred"].astype(np.int64), sp["succ"].astype(np.int64)
    w = 1.0 / np.diff(sp["succ_indptr"])[src]
    P_mem = sparse.csr_matrix((w, (src, orb_all[dst])),
                              shape=(nt, quo["n_orbits"] + 1))

    def disagreements(part):
        gap = P_mem[:n] - P[part]
        return int((np.abs(gap.data) > 1e-12).sum())

    assert disagreements(orbit_of) == 0
    moved = orbit_of.copy()
    moved[12345] = (moved[12345] + 1) % quo["n_orbits"]
    assert disagreements(moved) > 0, "the lumpability check is vacuous"
    split = orbit_of.copy()
    big = np.bincount(orbit_of).argmax()
    members = np.flatnonzero(orbit_of == big)
    split[members[: members.size // 2]] = quo["n_orbits"] - 1
    assert disagreements(split) > 0, "the lumpability check is vacuous"

    rng = np.random.default_rng(gf.SEED)
    B, C, interior = gf.draw_goal(rng, quo["absorbing"])
    q, _, _ = gf.committor(P, B, C, interior)
    y = q[interior]
    idx = np.flatnonzero(interior)
    Pi = sparse.csr_matrix(P[idx])
    q1 = np.asarray(Pi @ B.astype(float)).ravel()
    leak = np.zeros(P.shape[0])
    leak[idx] = q1
    leak[C] = 1.0                       # the bug: paths through C are not killed
    assert float((q1 + np.asarray(Pi @ leak).ravel() - y).max()) > 1e-9,         "the q2 <= q check is vacuous"


def test_truncated_resolvents_are_nested_under_the_committor(small):
    """q1 <= q2 <= q on every interior state of every draw."""
    assert small["n_draws"] == 8
    assert small["max_q1_minus_q2"] <= 1e-12
    assert small["max_q2_minus_q"] <= 1e-12
    assert small["max_committor_residual"] < 1e-9


def test_the_pooled_shallow_fit_does_not_clear_the_kill_threshold(small):
    """The refutation the proposer aimed at his own idea, at 8 draws."""
    assert gf.KILL_R2 == 0.95
    assert small["r2_linear"] == pytest.approx(SMALL["r2_linear"], abs=5e-4)
    assert small["r2_linear"] < gf.KILL_R2
    assert small["r2_linear_within"] == pytest.approx(SMALL["r2_linear_within"],
                                                      abs=5e-4)


def test_most_of_a_pooled_r2_is_free(small):
    """Predicting each draw's own MEAN, with no state skill at all, already
    scores this much pooled R2. Any pooled number read without it is inflated."""
    assert small["marginal_pooled"] == pytest.approx(SMALL["marginal_pooled"],
                                                     abs=5e-4)
    assert small["marginal_pooled"] > 0.5


def test_headroom_is_positive_over_the_three_ceilings(small):
    assert small["ceiling_marginal"] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= small["ceiling_lookup"] <= 1.0
    assert 0.0 <= small["ceiling_heuristic"] <= 1.0
    assert small["headroom"] == pytest.approx(SMALL["headroom"], abs=5e-4)
    assert small["headroom"] > 0.5


def test_the_depth_d_dead_mass_is_read_off_the_labels(small):
    """Where B' is more than d hops away the d-hop truncation is EXACTLY zero
    while q is not. Two independent routes to that fraction -- the truncation
    itself and a BFS on the move graph -- and they must agree."""
    for d in (1, 2, 4, 8):
        assert small["dead_mass"][d] <= small["hop_uniform"][d] + 1e-9
        assert small["dead_mass"][d] >= small["hop_uniform"][d] - 0.30
    assert small["dead_mass"][1] > small["dead_mass"][2] > small["dead_mass"][8]


def test_the_pinned_hundred_draw_run_reproduces():
    """The headline. Re-derived here, never copied from the docstring: the run
    is executed and every published figure is compared against it.

    ~5 minutes. That is the price of a number that is bound rather than quoted,
    and DCM-1's +0.0935 is what the alternative costs.
    """
    r = gf.report(n_test=gf.N_TEST, n_train=gf.N_TRAIN, seed=gf.SEED)
    pin = gf.RESULTS
    assert r["n_orbits"] == pin["n_orbits"]
    assert r["n_rows"] == pin["n_rows"]
    assert r["dead_columns"] == ["in_check"]
    assert r["design_rank"] == pin["design_rank"], "27 features are not 27 directions"
    for k in ("r2_linear", "r2_quadratic", "r2_linear_within",
              "r2_quadratic_within", "r2_linear_heldout",
              "r2_linear_heldout_within", "marginal_pooled", "ceiling_lookup",
              "ceiling_heuristic", "headroom", "headroom_worst_draw"):
        assert r[k] == pytest.approx(pin[k], abs=5e-4), (k, r[k], pin[k])
    assert r["draws_with_headroom_below_0p2"] == pin["draws_with_headroom_below_0p2"]
    for d in (1, 2, 4, 8):
        assert r["dead_mass"][d] == pytest.approx(pin["dead_mass"][d], abs=5e-4)
        assert r["hop_uniform"][d] == pytest.approx(pin["hop_uniform"][d], abs=5e-4)
    assert r["r2_linear"] < gf.KILL_R2, "the leap would be DEAD"
    assert r["r2_quadratic"] < gf.KILL_R2
    assert r["r2_linear_heldout"] < gf.KILL_R2


def test_demo_exits_zero():
    """L-SURFACE: the exit code is asserted, not read through a pipeline."""
    p = subprocess.run(
        [sys.executable, "-m", "ceqjepa.goal_family", "--draws", "4", "--train", "3"],
        capture_output=True, text=True,
    )
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    assert "POOLED R2" in p.stdout
    assert "KILL LINE" in p.stdout
