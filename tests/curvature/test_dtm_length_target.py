"""Is exact distance-to-mate a length-generalisation target: encoder-independent
AND fully supplied at every length?

WHY THIS FILE EXISTS. Two beds carry the length question and each is broken in a
different way. ceqjepa/pi_jepa.py scores a target that MOVES WITH THE ENCODER --
its own docstring records target RMS 0.4226 untrained against 1.7368 trained --
so an absolute error is not comparable across arms. A human-move bed is
encoder-independent but its label supply is a fixed corpus, and the corpus thins
with length. ceqjepa/chess_steps.py holds a third thing neither of them has: an
exact, encoder-independent integer for 368,452 enumerated positions, generated
rather than harvested, so supply is a function of CPU seconds and not of a file.

WHAT THESE TESTS ASK, AND BOTH WERE WRITTEN RED. A target is only a LENGTH target
if a reader that sees a short suffix of the context does materially worse than a
reader that sees all of it. Distance-to-mate is a function of ONE board, so the
question is entirely about the sampler that produces the trajectory: how far back
the most recent move of the least-recently-moved piece lies. The module under
test enumerates positions and has no trajectory sampler at all, so it can neither
declare that distance nor guarantee supply at a requested length.

RUN: python -m pytest tests/curvature/test_dtm_length_target.py -v

REPRICED BY AUDIT, 2026-09-14, at commit 391a2d0 on WIN-16QAL06O9GB. The two
failures below are honest and reproduce. Struck from the report that shipped
them, for having no producer: the trajectory and ply throughput rates, the
one-time enumeration timings, and the survival fractions, which contradicted
the surviving producers three ways. Struck for the opposite sign: a suffix-vs-
whole-context table reporting last-32 as BETTER than the whole context, where
the assertion below measures it as worse by 0.0012. Struck as a green that
never happened: the claim that a re-weighted sampler turns the first test GREEN
-- no green event exists and this file still ships two failures.

The supply figure reported alongside these tests was also the wrong measurement
of contamination. The producer that printed the 0.00% state-suffix overlaps
printed, two lines above them, that 35.80% of holdout endpoint D4 orbits also
occur in train -- and distance-to-mate is constant on every orbit, so a seen
orbit hands over the label. Any split on this target is orbit-disjoint or it
leaks.
"""

import numpy as np
import pytest

chess_steps = pytest.importorskip("ceqjepa.chess_steps")
skl = pytest.importorskip("sklearn.ensemble")

L = 128                 #: the 8x arm of ceqjepa/t_length.py's own length ladder
WINDOW = 32             #: the shortcut reader's suffix, a quarter of the context
MIN_GAP = 0.05          #: nrmse the shortcut must LOSE by, or the bar is vacuous
N_WALKS = 40000
SEED = 0


def _uniform_walks(b, length, seed):
    """Uniform-random legal walks on the module's OWN successor CSR.

    Every step is a python-chess legal move by construction -- the CSR is built
    from `board.legal_moves` in chess_steps.space() and nothing here re-derives
    chess rules. Returns (endpoint, alive, per-ply mover, per-ply squares).
    """
    sp, o = chess_steps.space(), chess_steps.oracle()
    n, sink, keys, dtm = sp["n_positions"], sp["sink"], sp["keys"], o["dtm"]
    succ, ptr, deg = sp["succ"], sp["succ_indptr"], np.diff(sp["succ_indptr"])
    absorbing = np.zeros(sp["n_states"], bool)
    absorbing[:n][sp["no_move"]] = True
    absorbing[sink] = True

    rng = np.random.default_rng(seed)
    cur = rng.choice(np.flatnonzero(dtm >= 1), b, replace=True)
    live = ~absorbing[cur]
    sq = np.empty((length + 1, b, 3), np.int16)
    sq[0] = keys[cur, :3]
    who = np.full((length, b), -1, np.int8)
    for t in range(length):
        c = np.minimum(cur, n - 1)
        d = np.maximum(deg[c], 1)
        nxt = np.where(live, succ[ptr[c] + (rng.random(b) * d).astype(np.int64)], cur)
        a, e = keys[c, :3].astype(int), keys[np.minimum(nxt, n - 1), :3].astype(int)
        who[t] = np.where(live & (nxt != cur),
                          np.where(nxt == sink, 1, np.argmax(a != e, axis=1)), -1)
        cur = nxt
        live = live & ~absorbing[cur]
        sq[t + 1] = keys[np.minimum(cur, n - 1), :3]
    return cur, live, who, sq


def _nrmse_of_a_reader(x, y, frac=0.75):
    """Held-out nrmse of a depth-6 gradient-boosted tree -- the SHALLOW baseline.

    Shallow on purpose: if a reader this cheap ties the full-context reader, no
    architecture can be measured by the difference between them.
    """
    ntr = int(frac * y.size)
    m = skl.HistGradientBoostingRegressor(max_depth=6, max_iter=300, random_state=0)
    m.fit(x[:ntr], y[:ntr])
    pred, held = m.predict(x[ntr:]), y[ntr:]
    return float(np.sqrt(((held - pred) ** 2).mean()) / held.std())


def test_the_default_sampler_leaves_no_room_between_a_suffix_and_the_whole_context():
    """A length target must PAY for the tokens it asks the reader to hold.

    RED FIRST, and this is the kill: under the only sampler the module's own
    graph offers -- uniform over legal moves -- a reader holding the last 32
    moves of a 128-move context reaches the same exact distance-to-mate as a
    reader holding all 128. The label is fine. The trajectory is what carries no
    length in it.
    """
    sp, o = chess_steps.space(), chess_steps.oracle()
    n, dtm = sp["n_positions"], o["dtm"]
    cur, live, who, sq = _uniform_walks(N_WALKS, L, SEED)
    sel = np.flatnonzero(live)
    y = dtm[np.minimum(cur[sel], n - 1)].astype(float)
    sel, y = sel[y >= 0], y[y >= 0]
    assert sel.size > 2000, "too few survivors to measure: %d" % sel.size

    def suffix_reader(k):
        x = np.full((sel.size, 6), -1.0)
        seen = np.zeros((sel.size, 3), bool)
        for t in range(L - 1, L - 1 - k, -1):
            w, board = who[t][sel], sq[t + 1][sel]
            for p in range(3):
                hit = (w == p) & ~seen[:, p]
                x[hit, 2 * p], x[hit, 2 * p + 1] = board[hit, p] % 8, board[hit, p] // 8
                seen[hit, p] = True
        return np.column_stack([x, seen.astype(float)])

    short = _nrmse_of_a_reader(suffix_reader(WINDOW), y)
    whole = _nrmse_of_a_reader(suffix_reader(L), y)
    assert short - whole >= MIN_GAP, (
        "a last-%d-move reader ties the whole %d-move context on exact "
        "distance-to-mate: nrmse %.4f against %.4f, gap %+.4f, required %+.4f "
        "(n=%d survivors of %d walks). The target is encoder-independent and "
        "fully supplied, and it still measures nothing about length, because "
        "the uniform sampler refreshes every piece inside the window."
        % (WINDOW, L, short, whole, short - whole, MIN_GAP, sel.size, N_WALKS))


def test_chess_steps_can_supply_trajectories_at_a_requested_length_and_lookback():
    """The reroute needs ONE surface the module does not have.

    RED FIRST. Enumeration gives positions; a length bar needs trajectories, at a
    requested length, at a requested count, with the sampler's LOOKBACK declared
    beside them -- the distance back to the most recent move of the
    least-recently-moved piece, which is the only thing that makes a suffix
    reader lose. Without it a caller cannot tell a length result from
    test_the_default_sampler_leaves_no_room_between_a_suffix_and_the_whole_context.
    """
    traj = getattr(chess_steps, "trajectories", None)
    assert traj is not None, (
        "ceqjepa.chess_steps exposes no `trajectories`. The route this file "
        "prices needs trajectories(n, length, lookback=..., seed=...) returning "
        "(states, moves, dtm, lookback_median) with exactly `n` labelled "
        "trajectories at `length` -- refilling absorbed walks rather than "
        "returning fewer -- and the achieved lookback beside them. "
        "dir() carries: %r" % sorted(a for a in dir(chess_steps) if not a.startswith("_")))
