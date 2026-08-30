"""DEIMOS / MORIARTY's moon, R9 iteration 1 -- the three surfaces Mars left

    "Unaudited: scale/e4_harmonic*, scale/eprocess_perdraw.py, most of
    ceq/nash.py."

CHARACTERISATION, NOT REPAIR. Every assertion below records a defect live in
the tree at bb0c04a. Each test PASSES today because the defect is present;
each would FAIL if the defect were fixed. Nothing here repairs anything and
nothing here touches an existing test.

Ranked by cost if true, highest first.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch


# =============================================================================
# ATTACK 1 -- eprocess.Eprocess.value overflows well inside the per-draw
# process's own designed operating range, on exactly the stream a positive
# result looks like.
# =============================================================================
def test_eprocess_value_overflows_deep_inside_its_own_designed_operating_range():
    """`scale/eprocess.py` keeps the mixture in LOG SPACE per the class
    docstring (`Eprocess`, `:294-301`): "G8: the decision compares a mixture
    value against a constant, and the underflow that would matter is an arm
    decaying to zero, which in log space decays linearly." That covers
    UNDERFLOW. It does not cover the opposite direction: `Eprocess.value`
    (`:311-313`) is `math.exp(_logsumexp(self.log_arm) - log(|grid|))`, called
    on EVERY `update()` (`:330`) to refresh `self.peak`. `_logsumexp` itself
    never overflows (it only ever exponentiates `x - max(x) <= 0`), but the
    FINAL `math.exp` call in `.value` re-materialises the whole accumulated
    log-evidence to a float, and that call has no such protection.

    `scale/eprocess_perdraw.py`'s entire reason to exist is pushing `t` from
    5 training seeds to `n_eval` draws -- 2048 per seed, 10240 pooled over the
    shipped `--seeds 0 1 2 3 4` default (`eprocess_perdraw.py:197`) -- and its
    `run()` (`:134-160`) calls `pair.update(d)` in an unconditional loop with
    NO early exit on `pair.decision` being set, only on a `void` ValueError.
    `eprocess.live()` (`:584-611`) does the same over its own `paired` list.

    So the ceiling that matters is not `log10_max_attainable`'s t=1748 (that
    guards the DIRECT, unused `max_attainable` formula) -- it is wherever
    `Eprocess.value`'s own `math.exp` first sees an argument past ~709. Below,
    a plausible winning stream -- mean 0.3 NRMSE margin on the clipped [-B,B]
    scale, sd 0.3, seed 0 -- crosses the decision THRESHOLD=40 at draw 67 (a
    real decision, exactly what the round wants to report) and then CRASHES
    the same object with `OverflowError` at draw 10135, one shy of the shipped
    pooled total of 10240. The instrument built to let a decision survive a
    slipped schedule instead survives the DECISION and then throws on the
    trailing draws.
    """
    from scale import eprocess as EP

    # -- reproduce the crash on the exact stream a strong, plausible arm
    #    would produce, at the exact shipped pool size (2048 * 5 seeds).
    rng = np.random.default_rng(0)
    n = 2048 * 5
    d = np.clip(rng.normal(0.3, 0.3, n), -EP.B, EP.B)

    e = EP.Eprocess()
    crossed_at = None
    with pytest.raises(OverflowError, match="math range error"):
        for i, di in enumerate(d, 1):
            v = e.update(float(di))
            if crossed_at is None and v >= EP.THRESHOLD:
                crossed_at = i

    assert crossed_at is not None and crossed_at < 100, (
        "sanity: the stream must actually cross the decision threshold long "
        f"before it crashes, got crossed_at={crossed_at}")

    # -- second, independent path: a hand-adversarial stream (d = +B always)
    #    crashes far sooner (draw 1757), which is the SMALLEST case that must
    #    fail if the log-space protection were actually complete end to end.
    e2 = EP.Eprocess()
    with pytest.raises(OverflowError):
        for _ in range(3000):
            e2.update(EP.B)

    # -- the calibration battery that is supposed to vouch for this
    #    construction (`calibrate`/`_run_block`, :383-437) NEVER calls
    #    `Eprocess`/`Pair` at all -- it independently re-derives the same
    #    product with `np.cumsum(np.log1p(...))` and compares in LOG SPACE
    #    (`ls >= log_thr`, :420-421), which is exactly why it never
    #    exponentiates a large log-evidence and never sees this crash. The
    #    branch actually used on real data (`Eprocess.update`) never executes
    #    inside the must-fire battery that is supposed to certify it.
    import inspect
    calibrate_src = inspect.getsource(EP.calibrate) + inspect.getsource(EP._run_block)
    assert "Eprocess" not in calibrate_src and "Pair(" not in calibrate_src, (
        "if this fires, calibrate() now exercises the real Eprocess class and "
        "this finding's second half (untested branch) no longer holds")


def test_eprocess_calibrate_max_peak_clamp_is_the_same_overflow_class_dormant():
    """`calibrate` (`:383-437`) computes `max_log = max(max_log, ls.max())`
    across replays and reports `max_peak = math.exp(min(max_log, 700.0))`
    (`:435`). The `min(..., 700.0)` is a SILENT clamp, not a raise: once the
    true peak log-evidence exceeds 700, the reported `max_peak` quietly reads
    a smaller, wrong number instead of the true one. It does not corrupt any
    verdict here -- `ok_null`/`ok_planted`/`ok_broken` all read `cross_*`
    booleans computed by comparing LOGS against `log_thr`, never `max_peak` --
    so this is diagnostic-only and DORMANT at the horizons actually shipped
    (`print_price` horizon=1500, `main()` default horizon=400): both stay
    under 700. It is the same overflow-avoidance-by-silent-clamp class as
    ATTACK 1's crash, one call away from mattering if a caller ever raises the
    horizon (the CLI accepts `--horizon` with no upper bound).
    """
    from scale import eprocess as EP

    # the true (unclamped) peak at a horizon already shipped in this file
    # (print_price's horizon=1500) is comfortably under the clamp -- verified
    # so this is filed as DORMANT, not as a live corruption.
    t = 1500
    true_max_log = EP._logsumexp(
        [t * math.log1p(lam) for lam in EP.LAMBDA_GRID]) - math.log(len(EP.LAMBDA_GRID))
    assert true_max_log < 700.0, "shipped horizon=1500 no longer safe; escalate"

    # but the clamp itself is real and silently wrong past it: construct the
    # same computation calibrate() does, at a horizon large enough to trip it,
    # and show the clamped output disagrees with the true value instead of
    # raising.
    t_over = 5000
    true_ln = EP._logsumexp(
        [t_over * math.log1p(lam) for lam in EP.LAMBDA_GRID]) - math.log(len(EP.LAMBDA_GRID))
    assert true_ln > 700.0
    with pytest.raises(OverflowError):
        math.exp(true_ln)                       # what calibrate() would do unclamped
    clamped = math.exp(min(true_ln, 700.0))      # what calibrate() actually reports
    assert clamped == pytest.approx(math.exp(700.0))
    assert math.log(clamped) < true_ln - 1000    # arbitrarily far below the truth,
                                                  # compared in log space since the
                                                  # true value itself is unrepresentable


# =============================================================================
# ATTACK 2 -- ceq.nash.nash_operator's default (tau=None) auto-safe path
# shares ONE tau across the whole batch, exactly the anti-pattern
# scale/negation_scope.py names and avoids for the same construction.
# =============================================================================
def test_nash_operator_shares_one_tau_across_the_whole_batch_and_overdamps_it():
    """`scale/negation_scope.py:462-466` documents the correct pattern for
    this exact construction: "`tau` is `ceq.nash.safe_tau` PER EXAMPLE.
    Calling `safe_tau` on the whole batch would return the max over it, which
    makes the label depend on the batch size; the per-example loop keeps the
    label a function of the example alone." `ceq/nash.py:146` does precisely
    the thing that sentence warns against:

        t = tau if tau is not None else safe_tau(game.reshape(-1, s_len, s_len))

    `game` already carries the batch dimension, so `reshape(-1, s_len, s_len)`
    is a no-op and `safe_tau`'s internal `.max()` (`:64`) takes the max over
    the ENTIRE batch. `ceq.arms.NashArm` (`ceq/arms.py:46,75`) is built with
    `tau=None` by default and never overridden in `run_all`/`train_one`, so
    this IS the path the benchmarked arm actually trains and evaluates on --
    not a corner case.

    Consequence, measured on the real corpus (`ceq.corpus.build`, n_train=384,
    seed=0, the exact construction `Arm.operator` uses for kind="nash"): the
    batch-shared tau is 2.2x the MEDIAN per-example safe_tau, so 56% of
    examples are damped far past what their own game would require, and mean
    |stance| -- the entire signed signal `ceq/nash.py`'s docstring says buys
    composition -- is cut roughly in half versus the per-example tau
    `negation_scope.py` uses for the same construction. `tests/w7/test_w7_nash.py`
    currently records the consequence at the score level: the nash arm's OOD
    NRMSE is 5.27, worse than predicting the mean (>1.0) and worse than the
    signed regression it was built to replace (3.68-3.77) -- run live in this
    session, `python -m pytest tests/w7/test_w7_nash.py -q` -> 4 failed
    (`test_nash_arm_generalizes_under_intervention`,
    `test_nash_arm_beats_the_signed_regression_it_replaces`, both device
    params). This does not prove the batch-shared tau is the whole cause, but
    it is A cause the module's own sibling file already knew to avoid.
    """
    import math as _math

    from ceq import arms, corpus, nash

    torch.manual_seed(0)
    data = corpus.build(n_train=384, n_test=128, seed=0)
    device = torch.device("cpu")
    arm = arms.build("nash", data["vocab"], device=device)
    assert arm.tau is None, "sanity: the benchmarked arm uses the auto-tau path"

    t = torch.tensor([r["tokens"] for r in data["train"]], dtype=torch.long,
                     device=device)
    x = arm.emb(t)
    query, key, value = arm.wq(x), arm.wk(x), x
    s_len, d = query.shape[-2], query.shape[-1]
    g = (value @ value.transpose(-2, -1)) / _math.sqrt(d)
    game = (g + g.transpose(-2, -1)) / 2.0

    batch_tau = nash.safe_tau(game.reshape(-1, s_len, s_len))
    per_ex_tau = torch.tensor([nash.safe_tau(game[i]) for i in range(game.shape[0])],
                              dtype=torch.float64)

    assert batch_tau == pytest.approx(float(per_ex_tau.max()), rel=1e-6), (
        "the batch-shared tau must equal the WORST-CASE example's own tau, "
        "which is the mechanism: safe_tau(reshape(-1,s,s)) is a global max")
    ratio_to_median = batch_tau / float(per_ex_tau.median())
    assert ratio_to_median > 1.5, (
        f"expected the batch tau to be well above the median per-example "
        f"tau, got ratio {ratio_to_median}")
    frac_overdamped = float((per_ex_tau < batch_tau / 2).float().mean())
    assert frac_overdamped > 0.3, (
        f"expected a large fraction of examples damped past half their own "
        f"safe_tau, got {frac_overdamped}")

    bias = value.mean(-1)
    stance_batch = 2.0 * nash.qre_stance(game, bias, tau=batch_tau,
                                         iters=nash.DEFAULT_ITERS) - 1.0
    stance_perex = 2.0 * torch.stack([
        nash.qre_stance(game[i], bias[i], tau=float(per_ex_tau[i]),
                        iters=nash.DEFAULT_ITERS)
        for i in range(game.shape[0])]) - 1.0

    assert float(stance_batch.abs().mean()) < 0.6 * float(stance_perex.abs().mean()), (
        "expected the batch-shared tau to roughly halve the mean stance "
        "magnitude relative to the per-example tau negation_scope.py uses")


# =============================================================================
# ATTACK 3 -- qre_stance's own safety mechanism (safe_tau / return_residual)
# is never consumed outside nash.py's default branch; explicit-tau call
# sites (including the module's own unit tests) run unprotected, and a real
# below-threshold game demonstrably has multiple equilibria.
# =============================================================================
def test_qre_residual_is_reported_but_never_read_anywhere_in_the_repo():
    """`ceq/nash.py:75-79` -- "`qre_stance` reports its own residual rather
    than asserting convergence... because a stance read off a non-converged
    iterate is not an equilibrium." That is the mitigation. Grepping the
    live tree for the ONLY way a caller can obtain that residual --
    `return_residual=True` -- finds it nowhere except the two lines inside
    `qre_stance` itself: the parameter default and the branch that uses it.
    Every call site in the repository (`nash_operator`'s own call at `:147`,
    `tests/w7/test_w7_nash.py`'s six direct calls) takes the default `False`
    and gets only `s`. The residual the docstring says exists to be read is
    dead code.
    """
    import pathlib
    import subprocess

    # SCOPED TO THE GIT-TRACKED SET, not a filesystem walk. A previous version
    # of this test used `root.rglob("*.py")`, which -- inside the primary repo
    # checkout, where every agent's worktree lives nested under
    # `.claude/worktrees/*` -- silently walked into every sibling worktree's
    # own copy of `ceq/nash.py` and over-counted (12 hits: 10 phantom copies
    # under `.claude/worktrees/`, this file quoting the name once, and the one
    # real file). "Does this symbol appear in this repository" means the
    # repository -- the git-tracked set -- not whatever nested checkouts
    # happen to sit on disk beneath it. See DEIMOS_REPORT.md, "A search that
    # over-finds is the same failure as one that under-finds, sign flipped."
    root = pathlib.Path(__file__).resolve().parents[2]
    self_rel = pathlib.Path(__file__).resolve().relative_to(root)
    tracked = subprocess.run(["git", "ls-files", "--", "*.py"], cwd=root,
                             capture_output=True, text=True, check=True
                             ).stdout.splitlines()
    hits = []
    for name in tracked:
        rel = pathlib.Path(name)
        if rel == self_rel:
            continue                                  # this file quotes the name
        text = (root / rel).read_text(encoding="utf-8")
        if "return_residual" in text:
            hits.append(rel)

    assert hits == [pathlib.Path("ceq/nash.py")], (
        f"return_residual is read somewhere outside its own definition now "
        f"({hits}); if so this finding is stale and the residual is consumed")


def test_nothing_enforces_tau_above_tau_star_at_explicit_call_sites():
    """`safe_tau` computes `tau*`; nothing compares a caller-supplied `tau`
    against it. `tests/w7/test_w7_nash.py` hardcodes `tau=0.25` twice
    (`:109-110,123`) and `tau=0.5` twice (`:135,146`) with no check against
    `safe_tau` for the game in question. Measured here for the EXACT
    construction those tests use (`m=(m+m.T)/2/sqrt(n)`, n=24, seed=0): the
    hardcoded `tau=0.25` sits at 0.749x `tau*` -- inside the region the
    module's own docstring says can have several equilibria and an iterate
    that "need not converge to any particular one."
    """
    from ceq import nash

    def game(seed=0, n=24, dtype=torch.float64):
        g = torch.Generator(device="cpu").manual_seed(seed)
        m = torch.randn(n, n, generator=g, dtype=dtype)
        m = (m + m.T) / 2 / n ** 0.5
        b = torch.randn(n, generator=g, dtype=dtype)
        return m, b

    m, b = game(seed=0, n=24)
    tau_star = nash.safe_tau(m, margin=1.0)
    assert 0.25 < tau_star, (
        "the w7 test suite's hardcoded tau=0.25 must sit below tau* for this "
        "to be the unprotected regime; if this fails the finding is stale")
    assert 0.25 / tau_star < 0.9


def test_below_tau_star_a_real_game_has_multiple_equilibria_from_different_starts():
    """The docstring's warning made concrete: a symmetric potential game with
    `tau` well below its own `tau*` (0.168x, margin=1.0) has at least TWO
    genuine fixed points -- both residual 0.0, i.e. both real solutions of
    `s = sigmoid((Ms+b)/tau)`, not iteration artifacts -- reached from
    different (fixed, non-adversarial) starting points. `qre_stance` always
    starts at the barycentre, so THIS particular call is deterministic, but
    nothing tells a caller that a different, equally valid equilibrium exists
    a `tau*`-sized step away, and nothing checks that the barycentre's
    equilibrium is the one anyone wants.
    """
    from ceq import nash

    seed, n, scale = 159, 6, 8.0
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = torch.randn(n, n, generator=g, dtype=torch.float64)
    m = (m + m.T) / 2 * scale
    b = torch.randn(n, generator=g, dtype=torch.float64) * 0.5
    tau = 1.0
    tau_star = nash.safe_tau(m, margin=1.0)
    assert tau < tau_star, "fixture drifted; must be below the contraction bound"

    def run_from(start, iters=5000):
        s = start.clone()
        for _ in range(iters):
            s = torch.sigmoid((m @ s + b) / tau)
        resid = float((s - torch.sigmoid((m @ s + b) / tau)).abs().max())
        return s, resid

    s_bary, r_bary = run_from(torch.full((n,), 0.5, dtype=torch.float64))   # qre_stance's own start
    s_zero, r_zero = run_from(torch.zeros(n, dtype=torch.float64))
    s_one, r_one = run_from(torch.ones(n, dtype=torch.float64))

    assert max(r_bary, r_zero, r_one) < 1e-8, "all three must be genuine fixed points"
    diff_zero = float((s_bary - s_zero).abs().max())
    diff_one = float((s_bary - s_one).abs().max())
    assert max(diff_zero, diff_one) > 0.5, (
        f"expected a different start to land on a materially different "
        f"equilibrium below tau*, got max diff {max(diff_zero, diff_one)}")


# =============================================================================
# ATTACK 4 -- tests/cameron/test_harmonic_attribution.py: an entire "U1/N3"
# contract clause, with specific pilot numbers, whose implementation does not
# exist anywhere in the tree.
# =============================================================================
def test_harmonic_attribution_battery_cites_a_producer_that_does_not_exist():
    """`tests/cameron/test_harmonic_attribution.py` is the sole integration
    surface between `scale/e4_harmonic.py`'s absorbing-chain machinery
    (`EH.case_graph`, `EH.absorbing_chain`, `EH.fixed_point`,
    `EH.hop_reading`, all real and imported at its `:78`) and a "U1/N3"
    contract clause it describes in detail: a harmonic-measure kernel, a
    masking-displacement probe, a rank cross-check, and a
    `PREREGISTERED_RHO_FLOOR` frozen from named pilot numbers ("spearman
    (omega_rank, mean_displacement_rank) = 0.743864, bootstrap CI
    [0.656532, 0.816955]", `:47-48`) -- written with the same provenance
    style as every genuinely-measured entry in this repo.

    None of the nine names the file calls on `scale.negation_scope` exist.
    This is P-2 (`MISTAKES.md`) with no floor: P-2's instances at least have a
    commit message as a producer. Here there is no producer at all -- the
    pilot numbers above have no possible source in the current tree. Every one
    of the file's 11 tests fails with `AttributeError`, confirmed by running
    it live this session (`python -m pytest tests/cameron/test_harmonic_attribution.py -q`
    -> 11 failed), all fresh `AttributeError`s and not a symptom of the
    128/146-failure baseline this repo already carries.
    """
    from scale import negation_scope as NS

    missing = ["absorbing_boundary_kernel", "harmonic_measure",
               "harmonic_label_batch", "train_control_arm",
               "masking_displacement", "rank_crosscheck", "dead_control_arm",
               "u1_attribution_run", "PREREGISTERED_RHO_FLOOR"]
    present = [name for name in missing if hasattr(NS, name)]
    assert present == [], (
        f"if this fires, {present} now exist on negation_scope and the "
        f"harmonic-attribution battery may actually be runnable; re-check "
        f"tests/cameron/test_harmonic_attribution.py before treating it as live")


# =============================================================================
# ATTACK 5 -- scale/e4_harmonic.case_graph's lru_cache shares a mutable
# adjacency structure across every call with the same arguments.
# =============================================================================
def test_case_graph_lru_cache_shares_a_mutable_adjacency_list_of_sets():
    """`scale/e4_harmonic.py:115` decorates `case_graph` with
    `functools.lru_cache(maxsize=4)`, and the function returns
    `adjacency: list[set[int]]` (`:143`) -- a mutable structure. `lru_cache`
    returns the SAME object on every subsequent call with the same
    arguments. Nothing in `e4_harmonic.py` itself mutates the returned
    adjacency (confirmed by reading `absorbing_chain`, `local_features`,
    `cheeger_t_rel_floor`, `_bridge_side` -- all read-only), and no caller
    found in the current tree (`tests/cameron/test_e4_harmonic.py`,
    `tests/cameron/test_harmonic_attribution.py`, both via
    `EH.case_graph(*TEST_CASE)`) mutates it either, so this is DORMANT, not
    presently corrupting any live reading. It remains a live footgun: any
    future caller that adds or removes an edge in place (the natural way to
    write a "what if this edge were cut" probe) silently corrupts the cached
    graph for every other caller in the process that draws the same case.
    """
    import scale.e4_harmonic as EH

    adj, nodes, bridge = EH.case_graph(*EH.SMALL_CASE[1:])
    sentinel = -999999
    assert sentinel not in adj[0]
    adj[0].add(sentinel)

    adj2, nodes2, bridge2 = EH.case_graph(*EH.SMALL_CASE[1:])
    assert adj2 is adj, "lru_cache must return the identical object"
    assert sentinel in adj2[0], (
        "a mutation to one caller's adjacency is visible to every other "
        "caller of case_graph with the same arguments")

    # clean up so this test does not itself poison the process-wide cache for
    # any other test file that also draws EH.SMALL_CASE in the same session
    adj[0].discard(sentinel)
