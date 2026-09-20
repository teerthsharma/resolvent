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
import math
import subprocess
import sys

import numpy as np
import pytest
from scipy import sparse

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


def test_the_frozen_random_control_is_never_trained(small):
    """THE MANDATORY CONTROL. `ceilings()` already supplies a trivial baseline
    ON THE TARGET (marginal/lookup/heuristic); the component ledger also wants
    one ON THE MODEL -- an arm at attn-2's own architecture and parameter
    count, weights never updated, eval-only. RED FIRST: before this arm
    existed `small["arms"]["frozen-random"]` was a KeyError, because grep found
    no frozen or untrained arm anywhere in this file."""
    a = small["arms"]["frozen-random"]
    b = small["arms"]["attn-2"]
    assert a["n_params"] == b["n_params"] and a["width"] == b["width"], (
        "the control is not parameter-matched to attn-2", a, b)
    assert math.isnan(a["loss_first"]) and math.isnan(a["loss_last"]), (
        "frozen-random carries a training-loss curve -- its weights moved", a)


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


def test_the_frozen_random_arm_clears_the_bar_and_misses_the_kill():
    """THE CONTROL'S OWN BAR AND KILL, at the size the ledger states them at:
    marginal 0.0000, lookup 0.2336 (`dr.RESULTS`). BAR: an untrained arm must
    not clear the marginal ceiling. KILL: it must not approach the lookup
    ceiling -- scoring near a trained arm there would mean the metric reads
    the lookup table by luck and not by training, voiding every depth-race
    number in the record. No attn arm needs training to answer this: depths=()
    and with_global=False skip every optimiser, so this reproduces in about a
    minute, not the headline run's 2,385.7s."""
    r = dr.race(n_test=100, n_train=20, seed=gf.SEED, depths=(), with_global=False)
    pin = dr.RESULTS
    assert r["ceilings"]["marginal"] == pytest.approx(pin["ceilings"]["marginal"],
                                                       abs=5e-4)
    assert r["ceilings"]["lookup"] == pytest.approx(pin["ceilings"]["lookup"],
                                                    abs=5e-4)
    a = r["arms"]["frozen-random"]
    assert a["n_params"] == pin["attn_params"][2], "not param-matched to attn-2"
    # re-derived, not copied: pinned so a silent change in init/wiring is caught
    assert a["within"] == pytest.approx(-5.4769, abs=5e-3), (
        "frozen-random's within-draw R2 moved -- re-check the bar and kill",
        a["within"])
    assert a["within"] <= r["ceilings"]["marginal"] + 1e-6, (
        "BAR FAILED: an untrained arm cleared the marginal ceiling", a["within"])
    assert a["within"] < r["ceilings"]["lookup"], (
        "KILL: frozen-random reached the lookup ceiling -- the metric is not "
        "measuring training, and every depth-race number above is void",
        a["within"], r["ceilings"]["lookup"])


def test_the_resume_guard_refuses_a_different_run(tmp_path):
    """R2-A, RED FIRST. At 62cb8e0 the resume guard asserted
    `order_hash == int(order.sum())`; `order` is a permutation of a fixed
    multiset `np.tile(np.arange(n_train), epochs)`, so its sum is the
    constant `epochs*n_train*(n_train-1)/2` for every seed and every `pool`
    -- the guard could not see either change, and a manual check against
    this file's own code (before this fix) resumed a seed=1 call silently
    into a seed=0 checkpoint. Fixed to key on (seed, n_train, epochs, pool)
    plus a CRC of the actual order bytes, asserted on load."""
    ckpt = str(tmp_path / "run")
    r0 = dr.race(n_test=3, n_train=3, epochs=2, seed=0, ckpt=ckpt,
                 depths=(2,), with_global=False)

    with pytest.raises(AssertionError, match="resumed a different run"):
        dr.race(n_test=3, n_train=3, epochs=2, seed=1, ckpt=ckpt,
                depths=(2,), with_global=False)

    with pytest.raises(AssertionError, match="resumed a different run"):
        dr.race(n_test=3, n_train=3, epochs=2, seed=0, ckpt=ckpt,
                depths=(2,), with_global=False, pool=5)

    # the SAME run (same seed, n_train, epochs, pool) still resumes, and the
    # checkpoint was already complete, so this re-scores the identical net
    # and reaches the identical loss -- the file's own :197-199 claim.
    r1 = dr.race(n_test=3, n_train=3, epochs=2, seed=0, ckpt=ckpt,
                 depths=(2,), with_global=False)
    assert r1["arms"]["attn-2"]["within"] == pytest.approx(
        r0["arms"]["attn-2"]["within"], abs=1e-9)
    assert r1["arms"]["attn-2"]["train_within"] == pytest.approx(
        r0["arms"]["attn-2"]["train_within"], abs=1e-9)


def test_the_cli_exposes_what_race_accepts(monkeypatch):
    """RED FIRST: at 62cb8e0 `depth_race.py --pool 5` failed argparse itself --
    `error: unrecognized arguments: --pool 5` -- because the parser never
    declared the flag, though `race()` has always taken `pool`. Same story for
    `ckpt`, `cache` and `train_stacks`. This does not train anything: `race` is
    replaced so the assertion is on the wiring between the parser and the
    call, not on a run."""
    captured = {}

    def fake_race(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "floor": gf.FLOOR, "n_test": 0, "n_train": 0, "n_rows": 0,
            "ceilings": {"marginal": 0.0, "lookup": 0.0, "heuristic": 0.0},
            "ceiling_lookup_worst": 0.0, "ceiling_heuristic_worst": 0.0,
            "headroom_worst": 0.0, "hop_fraction": {d: 0.0 for d in dr.DEPTHS},
            "arms": {}, "one_solve_claim_retired": False, "seconds": 0.0,
        }

    monkeypatch.setattr(dr, "race", fake_race)
    dr._cli(["--draws", "1", "--train", "1", "--epochs", "1",
              "--pool", "5", "--ckpt", "scratch/ck", "--cache", "scratch/c",
              "--no-train-stacks"])
    kw = captured["kwargs"]
    assert kw["pool"] == 5
    assert kw["ckpt"] == "scratch/ck"
    assert kw["cache"] == "scratch/c"
    assert kw["train_stacks"] is False

    # the untouched defaults still reach race() unchanged, so a run with none
    # of the new flags means exactly what it meant before this test existed
    captured.clear()
    dr._cli(["--draws", "1", "--train", "1", "--epochs", "1"])
    kw = captured["kwargs"]
    assert kw["pool"] is None and kw["ckpt"] is None and kw["cache"] is None
    assert kw["train_stacks"] is True


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


# ---------------------------------------------------------------------------
# V-12: A CONSTANT COMMITTOR IS A STRUCTURAL ARTIFACT, NOT A SMALL NUMBER
#
# VERDICT 2 (this lane's own investigation) traced the pool=160 crash to graph
# separation, not near-zero noise: 0 of 37,655/34,673/29,262 interior orbits
# can reach B' at all in test53/80/83, so q is constant on the interior
# IDENTICALLY, sd=0.0, not 1.11e-14. `committor()` (goal_family.py:591-631)
# now raises DegenerateLabelError at the solve -- the label's one builder, so
# one guard covers every one of its 20+ readers. What follows plants that
# structure on a 4-6 state toy chain instead of waiting to get unlucky on the
# real 46,137-orbit graph, and pins the two callers that must survive it:
# `goal_family.draw_test_goal` (already redraws, goal_family.py:557-560) and
# `depth_race.draws()`'s training pool (does not, depth_race.py:491).
# ---------------------------------------------------------------------------

def _toy_mask(n, *ix):
    m = np.zeros(n, dtype=bool)
    m[list(ix)] = True
    return m


def test_committor_rejects_a_planted_constant_goal_and_prints_its_sd():
    """PLANTED NEGATIVE. States 2 and 3 only cycle between each other and C --
    never toward B', in one hop or in any number of them -- so their committor
    is constant IDENTICALLY, the same structure VERDICT 2 found in the real
    chain. `committor` must reject it AT THE SOLVE, with the measured sd in
    the message: exactly 0.0, not a small residual, and printed rather than
    swallowed."""
    P = sparse.csr_matrix(np.array([
        [1.0, 0.0, 0.0, 0.0],   # 0 = B', absorbing
        [0.0, 1.0, 0.0, 0.0],   # 1 = C, absorbing
        [0.0, 0.5, 0.0, 0.5],   # 2 = interior: only reaches C and 3
        [0.0, 0.5, 0.5, 0.0],   # 3 = interior: only reaches C and 2
    ]))
    B, C, interior = _toy_mask(4, 0), _toy_mask(4, 1), _toy_mask(4, 2, 3)
    with pytest.raises(gf.DegenerateLabelError, match=r"sd=0\.000e\+00"):
        gf.committor(P, B, C, interior)


def test_lookup_ceiling_has_no_guard_of_its_own_the_builder_must_reject_first():
    """WHY THE FIX BELONGS IN THE BUILDER AND NOWHERE ELSE. `lookup_ceiling`
    computes `raw = 1 - SSE/SS` with no check on SS at all; handed a
    degenerate label (SS=0) it divides by exactly zero. Verified here on a
    label that "slipped past" the builder, standing in for what every one of
    `lookup_ceiling`'s 20+ callers would hit if the guard at
    `goal_family.py:627-628` were ever moved out of `committor` and into one
    reader instead of all of them."""
    y = np.zeros(3)  # a degenerate label, as if it had slipped past committor
    train = [np.array([1.0, 0.0, 1.0, 0.0])]
    interior = _toy_mask(4, 1, 2, 3)
    with pytest.raises(ZeroDivisionError):
        gf.lookup_ceiling(train, interior, y)


def test_draw_test_goal_redraws_a_planted_degenerate_candidate_without_scoring_it(monkeypatch):
    """THE CORRECT CALLER, PINNED AS A REGRESSION GUARD. `draw_test_goal`
    already catches `DegenerateLabelError` and tries again
    (goal_family.py:557-560); this drives it with a planted first candidate
    that is degenerate by construction and a second that is not, and checks
    the degenerate one never reaches `lookup_ceiling` -- not "scores near
    zero", never called at all, because `committor` raises before returning
    a q for it."""
    N = 6
    P = sparse.lil_matrix((N, N))
    P[0, 0] = 1.0                              # 0 = B', absorbing
    P[1, 1] = 1.0                              # 1 = C, absorbing
    P[2, 3] = P[2, 1] = 0.5                    # 2 = bad interior: cycles + C
    P[3, 2] = P[3, 1] = 0.5                    # 3 = bad interior: cycles + C
    P[4, 0] = 1.0                              # 4 = good interior: -> B'
    P[5, 1] = 1.0                              # 5 = good interior: -> C
    P = P.tocsr()

    B = _toy_mask(N, 0)
    C_bad, interior_bad = _toy_mask(N, 1), _toy_mask(N, 2, 3)
    C_good, interior_good = _toy_mask(N, 1), _toy_mask(N, 4, 5)

    candidates = [(B, C_bad, interior_bad), (B, C_good, interior_good)]
    monkeypatch.setattr(gf, "draw_goal",
                        lambda rng, absorbing: candidates.pop(0))

    calls = []
    real_ceiling = gf.lookup_ceiling

    def spy(train, interior, y):
        calls.append((interior.copy(), y.copy()))
        return real_ceiling(train, interior, y)

    monkeypatch.setattr(gf, "lookup_ceiling", spy)

    train_q = [np.where(B, 1.0, 0.0)]
    _, _, interior, q, res, iters, ceil, tries = gf.draw_test_goal(
        rng=np.random.default_rng(0), absorbing=None, P=P, train_q=train_q,
        floor=1.0)

    assert (interior == interior_good).all(), (
        "the degenerate candidate was accepted, not redrawn", interior)
    assert tries == 2, "the first (degenerate) try must count against tries"
    assert len(calls) == 1, (
        "lookup_ceiling must be reached exactly once, for the good draw only",
        calls)
    assert calls[0][1].std() > 0, (
        "lookup_ceiling was handed a zero-variance label", calls[0][1])


def test_draws_training_pool_must_survive_a_degenerate_draw_like_report_does():
    """RED FIRST -- depth_race.py:491, `q, res, _ = gf.committor(P, B, C,
    interior)` in the training loop of `draws()`, has no
    `except DegenerateLabelError`, though `draws()`'s own docstring claims
    "the SAME draw stream `goal_family.report` uses" and `report()` (goal_
    family.py:806-810) redraws a degenerate TRAINING goal instead of dying on
    it. With the V-12 guard live this is now false for exactly the training
    goal that lands on a degenerate draw: `report` redraws, `draws` crashes.
    No real draw in the published record triggers it (0 degenerate goals
    among the first 160 training draws, seed 0), so `gf.committor` is
    monkeypatched to raise on its first call -- the same
    `DegenerateLabelError` VERDICT 2 measured on the real graph (sd=0.0, not a
    residual) -- forcing exactly the code path a real unlucky draw would hit.

    Captured verbatim, commit 1d7863f on WIN-16QAL06O9GB (python 3.11.9,
    numpy 2.4.6, scipy 1.17.1), calling `dr.draws(2, 2, gf.SEED, gf.FLOOR,
    pool=2)` with the first `gf.committor` call forced degenerate:

        ceqjepa.goal_family.DegenerateLabelError: committor is constant on
        23974 interior orbits (sd=0.000e+00 < LABEL_SD_MIN=1e-09): this
        draw's B'/C leave the interior no way to disagree, so its label is
        undefined for R^2, not small -- redraw, do not score it

    raised out of `ceqjepa/depth_race.py:491` (`draws`), uncaught -- the exact
    sibling gap VERDICT 2 named. This test wants NO raise and a completed
    pool of the requested size, exactly what `report()` already delivers for
    the same forced failure."""
    calls = {"n": 0}
    real_committor = gf.committor

    def flaky(P, B, C, interior, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise gf.DegenerateLabelError(0.0, int(interior.sum()))
        return real_committor(P, B, C, interior, *a, **kw)

    gf.committor = flaky
    try:
        train, test = dr.draws(2, 2, gf.SEED, gf.FLOOR, pool=2)
    finally:
        gf.committor = real_committor

    assert len(train) == 2, "the training pool must reach its requested size"
    assert len(test) == 2
    assert calls["n"] >= 3, (
        "the forced failure must have been redrawn, not merely swallowed",
        calls["n"])
