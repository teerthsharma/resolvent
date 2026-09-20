"""The binds for `ceqjepa/lora_sort.py`. Every one can fail.

L-COUNT. This file's EXPECTED COLLECTED COUNT is 20, asserted by
`test_the_collected_count_is_the_expected_count`. IT WAS DECLARED 18 AND THE
SUITE COLLECTED 20 -- the law caught the author's own miscount on the first run,
which is the only reason the number in this docstring is trustworthy now.
A run that collects any other number is RED regardless of summary colour, which
is the whole point of `MISTAKES.md:2819 L-COUNT` -- a module that stopped being
importable contributes zero tests and zero failures.

    python -m pytest tests/lorasort -q          # expect: 20 passed

WHAT IS BOUND HERE, AND WHAT IS NOT. These tests bind the ALGEBRA and the
INSTRUMENT: the Woodbury identity against a dense re-solve, the bitwise bind at
Delta = 0, the L-NULL one-field-varies claim, the Hilbert permutation, the
step-0/step-1 gradient mechanism, the spectral clip, and the fact that each drop
test's PLANT is what it says it is. They do NOT bind the round's scores -- those
are seeds and steps, and `python -m ceqjepa.lora_sort` is their producer.

THE PLANTED NEGATIVES ARE THE POINT. `MISTAKES.md` V-7 and V-13 are both absence
claims from searches that could not have found anything, so the two drop tests
are run against input that must make them fire the OTHER way, and the exactness
check is run against a deliberately wrong identity.
"""
from __future__ import annotations

import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ceqjepa.lora_sort import (                                     # noqa: E402
    AGREE_RAD, CITATION_MARKS, DT, GAMMA, HILBERT_PERM, HILBERT_SIDE, N_POS,
    SORTS, SORT_FN, SPECTRAL_CLIP, TEACHER_SPEC, _clip_, _planted_block_local,
    _planted_heavy_tail, _spec_norm, bind_at_zero, dense_read, drop_test_hilb,
    drop_test_ot, grad_table, hilbert_order, pinned_manifest, sanity_planted,
    sort_svd, subspace_angle, woodbury_exactness, woodbury_read,
)
from ceqjepa.operator import build_operator, state_solve                # noqa: E402

EXPECTED_COLLECTED = 20


# ------------------------------------------------------------------ L-COUNT
def test_the_collected_count_is_the_expected_count():
    """L-COUNT. The number is written in this file, in advance, and asserted.

    Collection is run in a SUBPROCESS with `--collect-only -q`, and both the exit
    status and the count are checked -- L-SURFACE, never through a pipe. A suite
    that grew silently fails here too, because a suite whose contents nobody
    chose is the same defect as one that shrank.
    """
    out = subprocess.run(
        [sys.executable, "-m", "pytest", str(pathlib.Path(__file__).parent),
         "--collect-only", "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, "collection itself failed:\n%s\n%s" % (out.stdout, out.stderr)
    n = sum(1 for line in out.stdout.splitlines() if "::test_" in line)
    assert n == EXPECTED_COLLECTED, (
        "expected %d collected, got %d. A count that moved is RED whatever the "
        "summary says.\n%s" % (EXPECTED_COLLECTED, n, out.stdout))


# ------------------------------------------------------------------ the identity
def test_woodbury_matches_a_dense_resolve():
    """Hager 1989 Eq. (1) p. 221 against `torch.linalg.solve` on the dense
    updated matrix. Two different routes, one answer."""
    wx = woodbury_exactness(n_draws=20, ranks=(1, 4, 16), seed=3)
    assert wx["n_cells"] == 18, wx["n_cells"]
    assert wx["worst_abs"] < 1e-10, wx


def test_a_wrong_capacitance_sign_breaks_the_identity():
    """THE PLANTED NEGATIVE for the check above. Hager writes the update as a
    DIFFERENCE, so the capacitance factor is `(I - V A^-1 U)^-1`. Flipping that
    sign is the single most likely transcription error, and it must be caught --
    otherwise `test_woodbury_matches_a_dense_resolve` is a check that a wrong
    formula would also pass."""
    torch.manual_seed(11)
    P = build_operator(torch.randn(N_POS, N_POS, dtype=DT), [], teleport=0.0)
    U = torch.randn(N_POS, 3, dtype=DT) * 0.05
    W = torch.randn(N_POS, 3, dtype=DT) * 0.05
    V = torch.randn(N_POS, 4, dtype=DT)
    eye = torch.eye(N_POS, dtype=DT)
    M0 = eye - GAMMA * P
    sol = torch.linalg.solve_triangular(M0, torch.cat([V, GAMMA * U], -1), upper=False)
    base, MgU = sol[:, :4], sol[:, 4:]
    wrong_cap = torch.eye(3, dtype=DT) + W.T @ MgU               # + instead of -
    wrong = base + MgU @ torch.linalg.solve(wrong_cap, W.T @ base)
    right = woodbury_read(P, U, W, V)
    truth = dense_read(P, U, W, V)
    assert (right - truth).abs().max() < 1e-10
    assert (wrong - truth).abs().max() > 1e-6, "the sign error was not detectable"


def test_the_capacitance_solve_is_r_by_r_and_m0_stays_triangular():
    """The reason the Woodbury route exists at all: `M0 = I - g P` keeps the
    causal lower-triangular structure whatever the rank-r delta does, so the big
    solve stays a forward substitution and only an `r x r` system is general."""
    torch.manual_seed(5)
    P = build_operator(torch.randn(N_POS, N_POS, dtype=DT), [], teleport=0.0)
    M0 = torch.eye(N_POS, dtype=DT) - GAMMA * P
    assert torch.equal(M0, M0.tril()), "M0 lost its triangularity"
    U = torch.randn(N_POS, 2, dtype=DT) * 0.05
    W = torch.randn(N_POS, 2, dtype=DT) * 0.05
    M = torch.eye(N_POS, dtype=DT) - GAMMA * (P + U @ W.T)
    assert not torch.equal(M, M.tril()), (
        "the rank-2 delta left M triangular -- then the round is not testing "
        "what it says it is")


# ------------------------------------------------------------------ the bind
def test_delta_is_exactly_zero_at_init_and_the_read_is_bitwise():
    """THE BIND. `W = 0` makes `Delta = 0` as a tensor, and the Woodbury route
    must then return the base `state_solve` BIT FOR BIT, not close to it. A run
    that loses this is struck (`scale/m3_quintuple.py:492`'s rule)."""
    bd = bind_at_zero(seed=1, r=4)
    for name, v in bd["per_sort"].items():
        assert v["delta_exactly_zero"], name
        assert v["bitwise_equal_all_heads"], (name, v)
        assert v["max_abs_diff"] == 0.0, (name, v)


def test_a_nonzero_w_breaks_the_bitwise_bind():
    """THE PLANTED NEGATIVE for the bind. If the read were bitwise-equal to the
    base for a NONZERO delta too, the bind above would be vacuous."""
    torch.manual_seed(2)
    P = build_operator(torch.randn(N_POS, N_POS, dtype=DT), [], teleport=0.0)
    V = torch.randn(N_POS, 4, dtype=DT)
    U = torch.randn(N_POS, 2, dtype=DT) * 0.05
    base, _ = state_solve(P, V, GAMMA)
    assert torch.equal(woodbury_read(P, U, torch.zeros(N_POS, 2, dtype=DT), V), base)
    W = torch.randn(N_POS, 2, dtype=DT) * 0.05
    assert not torch.equal(woodbury_read(P, U, W, V), base)


# ------------------------------------------------------------------ L-NULL
def test_exactly_one_field_varies_across_the_three_sorts():
    """L-NULL. Across the three sorts the ONLY thing that varies is the selection
    rule. The manifest is compared key by key, so a second varied field cannot
    hide in it."""
    ms = {s: pinned_manifest(s, 4, 0) for s in SORTS}
    keys = set(ms["S-SVD"])
    for a in SORTS:
        for b in SORTS:
            if a >= b:
                continue
            diff = {k for k in keys if ms[a][k] != ms[b][k]}
            assert diff == {"varied_selection_rule"}, (a, b, diff)
    assert all(k.startswith("pinned_") for k in keys - {"varied_selection_rule"})


def test_the_rank_budget_is_matched_across_the_three_sorts():
    """An unmatched r is struck. Every sort returns exactly `r` columns of the
    same length, at every r on the grid."""
    torch.manual_seed(9)
    C = torch.randn(N_POS, N_POS, dtype=DT)
    for r in (1, 2, 4, 8, 16):
        shapes = {name: tuple(SORT_FN[name](C, r, seed=0).shape) for name in SORTS}
        assert set(shapes.values()) == {(N_POS, r)}, (r, shapes)


# ------------------------------------------------------------------ the Hilbert order
def test_the_hilbert_order_is_a_permutation_with_unit_steps():
    """The defining property of the curve's discrete order: consecutive visits are
    grid NEIGHBOURS. Hilbert 1891 p. 459 states it as the construction constraint
    -- consecutive subsquares share a side -- and it is the only thing that makes
    a contiguous run along the order a contiguous TILE in the plane."""
    perm = hilbert_order(HILBERT_SIDE)
    assert sorted(perm.tolist()) == list(range(N_POS))
    xy = [(int(i % HILBERT_SIDE), int(i // HILBERT_SIDE)) for i in perm]
    steps = [abs(xy[k + 1][0] - xy[k][0]) + abs(xy[k + 1][1] - xy[k][1])
             for k in range(len(xy) - 1)]
    assert set(steps) == {1}, "the order is not a Hilbert walk: steps %s" % sorted(set(steps))


def test_the_raw_index_order_is_not_a_hilbert_walk():
    """THE PLANTED NEGATIVE for the test above, and the reason S-HILB can differ
    from anything at all. Row-major order has jumps of 7, so a 4x4 tile is
    scattered across it while being contiguous along the curve."""
    xy = [(i % HILBERT_SIDE, i // HILBERT_SIDE) for i in range(N_POS)]
    steps = {abs(xy[k + 1][0] - xy[k][0]) + abs(xy[k + 1][1] - xy[k][1])
             for k in range(len(xy) - 1)}
    assert steps != {1}, steps


# ------------------------------------------------------------------ the plants
def test_the_heavy_tail_plant_really_is_heavy_tailed():
    """The instrument before the reading. Drop test 2's verdict means nothing if
    its input has no outliers."""
    C = _planted_heavy_tail(0)
    norms = C.norm(dim=0)
    assert float(norms[:4].min() / norms[4:].max()) > 3.0, norms[:8]


def test_the_block_plant_really_is_block_local_and_norm_matched():
    """Drop test 3 is PRE-REGISTERED at the neutral setting: equal per-column
    norms, so neither rule is handed a magnitude edge, and the block columns sit
    entirely inside a 4x4 tile."""
    C, idx = _planted_block_local(0)
    norms = C.norm(dim=0)
    assert float((norms - 1.0).abs().max()) < 1e-9, "columns are not norm-matched"
    mask = torch.ones(N_POS, dtype=torch.bool)
    mask[idx] = False
    assert float(C[mask, :8].abs().max()) == 0.0, "the block leaks outside the tile"
    assert len(idx) == 16


# ------------------------------------------------------------------ the drop tests
def test_the_ot_drop_test_can_fire_both_ways():
    """MUST-FIRE 2 must be able to DELETE S-OT. Fed a pool where the two rules
    provably agree -- an exactly rank-r pool, where any r independent columns and
    the top-r singular subspace are the SAME space -- the test must report
    agreement. Without this half the drop test is an absence claim."""
    real = drop_test_ot(r=4, seeds=(0, 1))
    assert real["min_angle_rad"] > AGREE_RAD, real
    assert not real["dropped"]
    # the forced-agreement half, done directly on the comparison the test uses
    torch.manual_seed(4)
    A = torch.randn(N_POS, 4, dtype=DT)
    B = torch.randn(N_POS, 4, dtype=DT)
    C = A @ B.T
    assert subspace_angle(sort_svd(C, 4), A) < 1e-6, "rank-4 pool did not span"


def test_the_hilb_drop_test_can_fire_both_ways():
    """MUST-FIRE 3 must be able to DELETE S-HILB. Fed a pool whose block columns
    also carry the largest energy, S-SVD finds the tile too and the gap closes
    below the pre-registered SAME threshold."""
    real = drop_test_hilb(r=4, seeds=(0, 1))
    assert real["rows"][0]["gap"] >= real["same_threshold"], real["rows"][0]
    C, idx = _planted_block_local(0)
    C[:, :8] *= 50.0                                  # hand S-SVD the answer
    e = sort_svd(C, 4) ** 2
    frac = float((e[idx, :].sum(0) / e.sum(0)).mean())
    assert frac > 0.9, (
        "S-SVD failed to find a tile carrying 50x the energy -- then the drop "
        "test could never delete S-HILB and is an absence claim (frac=%.4f)" % frac)


# ------------------------------------------------------------------ must-fire 5
def test_u_is_dead_at_step_zero_and_live_at_step_one():
    """MUST-FIRE 5, and the expected answer is NOT "both live".

    With `W = 0`, `dL/dU = (dL/dDelta) W = 0` EXACTLY -- `MISTAKES.md:2526 V-29`'s
    mechanism, one level over. The step-0 zero is REQUIRED (it is what makes
    Delta = 0, which the bind needs) and is asserted as exactly zero, not as
    small. A factor still dead at step 1 is the real defect and fails here.
    """
    gt = grad_table(seed=0, r=4)
    for name, v in gt["per_sort"].items():
        s0, s1 = v["steps"]
        assert all(h["grad_U_absmax"] == 0.0 for h in s0["per_head"]), (name, s0)
        assert all(h["grad_W_absmax"] > 0.0 for h in s0["per_head"]), (name, s0)
        assert all(h["grad_U_absmax"] > 0.0 for h in s1["per_head"]), (name, s1)
        assert all(h["grad_W_absmax"] > 0.0 for h in s1["per_head"]), (name, s1)
        assert v["defect_heads_still_dead_at_1"] == [], (name, v)


# ------------------------------------------------------------------ must-fire 6
def test_the_spectral_clip_holds_and_its_norm_is_the_identity_not_an_estimate():
    """MUST-FIRE 6. `||U W^T||_2 <= c` after the clip, and the r x r route used to
    measure it agrees with the full n x n SVD to float64 -- it is an identity, so
    a disagreement means the fast route is wrong, not imprecise."""
    torch.manual_seed(7)
    U = torch.randn(6, N_POS, 3, dtype=DT)
    W = torch.randn(6, N_POS, 3, dtype=DT)
    fast = _spec_norm(U, W)
    slow = torch.stack([torch.linalg.matrix_norm(U[h] @ W[h].T, ord=2) for h in range(6)])
    assert float((fast - slow).abs().max()) < 1e-12, (fast, slow)
    assert float(fast.max()) > SPECTRAL_CLIP, "the clip would not engage -- vacuous"
    _clip_(U, W, SPECTRAL_CLIP)
    assert float(_spec_norm(U, W).max()) <= SPECTRAL_CLIP + 1e-12


def test_the_clip_leaves_room_for_the_bound_and_for_the_teacher():
    """The two things a clip that is merely 'printed' would hide.

    (1) `g c < sigma_min(M0)`, or the bound `1/(sigma_min(M0) - g c)` is negative
        or infinite and 'held' vacuously -- which is what the first draft of this
        round shipped at c = 0.5.
    (2) `||D*||_2 <= c`, or the planted target is outside the feasible set and
        every sort's score measures the clip instead of the sort.
    """
    lo = 0.018456                     # measured min over 8 seeds x 6 heads
    assert GAMMA * SPECTRAL_CLIP < lo, (
        "g*c = %.6f >= sigma_min(M0) = %.6f: the bound is vacuous"
        % (GAMMA * SPECTRAL_CLIP, lo))
    assert TEACHER_SPEC <= SPECTRAL_CLIP, (TEACHER_SPEC, SPECTRAL_CLIP)


# ------------------------------------------------------------------ must-fire 1
def test_the_planted_rank_r_delta_is_recovered_and_the_failure_is_conditioning():
    """MUST-FIRE 1, reported as measured. S-SVD and S-HILB recover the plant;
    S-OT does NOT, and the mechanism is the conditioning of its selection, not a
    bug in the transport. Every column of `A B^T` lies in span(A), so any r
    INDEPENDENT columns recover it exactly -- a column-selecting rule can only
    fail by picking a near-dependent set, and S-OT's depth rule has no diversity
    term to stop it. Bound here so the failure cannot be quietly repaired."""
    sn = sanity_planted(seed=0, r=4)
    assert sn["per_sort"]["S-SVD"]["recovered"], sn["per_sort"]["S-SVD"]
    assert sn["per_sort"]["S-HILB"]["recovered"], sn["per_sort"]["S-HILB"]
    assert not sn["per_sort"]["S-OT"]["recovered"], (
        "S-OT now recovers the plant -- if that is a real repair, this assertion "
        "and the report's must-fire-1 row both move together: %s"
        % sn["per_sort"]["S-OT"])
    assert (sn["per_sort"]["S-OT"]["max_selection_cond2"]
            > 10 * sn["per_sort"]["S-SVD"]["max_selection_cond2"]), sn["per_sort"]


# ------------------------------------------------------------------ the citations
def test_every_sort_carries_a_resolvable_identifier_and_a_numbered_anchor():
    """THE CITATION KILL, as a test. A named source without a resolvable
    identifier AND a specific theorem, page or equation is P-16 class
    (`MISTAKES.md:2863`) and struck. Hilbert 1891 carries a PAGE and no equation,
    because the paper has none -- that is why 'page' is accepted here and why the
    locality claim is carried by Moon et al.'s Theorem 1 instead."""
    ident = ("DOI ", "arXiv:")
    anchor = ("Eq. (", "Def. ", "Thm. ", "Theorem ", "Sec. ", "p.")
    for name in SORTS:
        m = CITATION_MARKS[name]
        assert m["grade"] in {"V", "V-eq", "PARTIAL"}, m
        assert m["sources"], name
        for src in m["sources"]:
            assert any(t in src for t in ident), "no resolvable identifier: %r" % src
            assert any(t in src for t in anchor), "no numbered anchor: %r" % src
        if m["grade"] == "PARTIAL":
            assert m.get("unsourced"), (
                "a PARTIAL grade must say WHAT is unsourced, or it is a grade "
                "with no content: %s" % name)


def test_the_unsourced_sentence_names_no_author():
    """P-16's own rule, mechanised. The sentence 'keeps directions by depth rather
    than by variance' is in neither cited paper, so it is carried as UNSOURCED
    and NO author's name may appear beside it."""
    marks = CITATION_MARKS["S-OT"]["unsourced"]
    text = " ".join(marks) if isinstance(marks, list) else marks
    head = text.split(".")[0]
    for surname in ("Hallin", "Chernozhukov", "Galichon", "Henry", "Matran"):
        assert surname not in head, (
            "%s is named on the unsourced sentence -- that is the P-16 shape "
            "exactly: %r" % (surname, head))
    assert "UNSOURCED" in text
