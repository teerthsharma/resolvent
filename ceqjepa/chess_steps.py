"""The chess bed: EXACT plies-to-loss by enumeration, and T-STEPS against its controls.

THE TRADE-OFF THIS MODULE REFUSES. The spec wanted an engine oracle. There is no
engine on this box -- no stockfish, no lc0, no tablebase -- and the repo's chess
labels are terminal results with no distance-to-terminal column. The stated
choice was "no engine, no oracle". It is a false one. The spec's other door is
ENUMERATION, and K+Q vs K is small enough to walk through completely: 368,452
legal positions, 4,891,672 legal moves, solved by backward induction over the
whole space. That is not an approximation of an engine. It is the ground truth an
engine approximates, and a tablebase for more pieces is the same computation
just larger.

THE ORACLE, and what makes it exact. Backward induction from the 364 checkmate
positions, layer by layer, until the fixpoint: a position is a WIN in d+1 plies
if some legal move reaches a LOSS in d, and a LOSS in d+2 if EVERY legal move
reaches a WIN in at most d+1. Nothing is capped and nothing is truncated -- the
induction runs until no label changes, in 10 layers. Positions the induction
never labels are DRAWS, and they are returned as Refusals carrying which kind,
never as zeros. Measured, `python -m ceqjepa.chess_steps`:

    max distance-to-mate, White to move    19 plies  = mate in 10 moves
    max distance-to-mate, Black to move    20 plies
    checkmates                                364
    stalemates                                872
    drawn by defence (the queen falls)     22,176
    win basin                             345,404    of 368,452 positions

19 plies is the published maximum for this endgame and this code did not produce
it -- landing on it exactly, from the other end, is the check. Two further
checks: a FORWARD alpha-beta minimax, sharing no code with the induction, agrees
on every sampled position at 5 plies or fewer; and three positions whose distance
is known from outside (a mate on the board, a mate-in-1, a forced mate-in-2 found
by exhaustive python-chess search) read 0, 1 and 2.

THE COMMITTOR is the transition-path-theory one and it is not the game value.
Under optimal play the committor to the losing basin is 0 or 1 and carries no
gradient at all, which makes it useless as a reaction coordinate; that
degeneracy is a property of a deterministic game, not of this bed. q_loss(s) is
therefore the probability that a walk from s reaches checkmate BEFORE a draw when
both sides move uniformly at random over their legal moves -- the committor of an
absorbing Markov chain, pinned to 1 on the 364 mates and to 0 on the 872
stalemates and on the K-vs-K state the queen's capture leads to. Solved as a
linear fixpoint, not simulated: ||q - Pq||_inf reaches 9.645e-13 in 1,085
iterations at Q_TOL = 1e-12, and 364,796 of the 368,452 positions carry an
interior value (max 0.2971, mean 0.0141).

THE PINNED CONSTANTS. Everything below is at

    ENDGAME       = "KQK"        the enumerated endgame
    SEED          = 0            every sample and every shuffle
    K_NEIGHBORS   = 8            inherited from ceqjepa.curvature, unchanged
    ALPHA         = 0.5          inherited from ceqjepa.curvature, unchanged
    N_ANCHOR + N_SCORED          364 basin anchors + 1636 scored positions
    EDGE_LENGTH   = "exp(-kappa)"

and any number quoted at other settings says so beside itself. W1 is not
computed here: kappa comes from ceqjepa.curvature.curvature_from_points, whose
W1 is the exact transport optimum. There is no second W1 in this file.

AN AFFINITY AND A LENGTH POINT IN OPPOSITE DIRECTIONS, and getting that backwards
silently inverts the result. An attention weight is HIGH where two nodes are
CLOSE; a path length is HIGH where they are FAR. Ollivier curvature is an
affinity-like quantity -- positive inside a dense region, negative on a bridge --
so mapping it onto a length needs a direction chosen on purpose and stated:

    EDGE_LENGTH          exp(-kappa)   a bridge (kappa < 0) is EXPENSIVE
    EDGE_LENGTH_INVERTED exp(+kappa)   a bridge (kappa < 0) is CHEAP

Both run as arms. exp is used rather than 1 - kappa because it is strictly
positive for every finite kappa, so no edge can be a negative cycle. Because
curvature is invariant under scaling the weights, raw path lengths are not
comparable across arms; every comparison here is in RANK terms (Spearman), which
is scale-free by construction.

T-STEPS, AND THE ANSWER, WHICH IS NO. The question was whether the geodesic on a
curvature-weighted context graph tracks exact plies-to-loss better than plain
Euclidean distance in the same embedding. Five arms, ONE encoding, ONE kNN graph,
ONE edge set of 10,125 edges over 2,000 nodes (364 basin anchors plus 1,636
scored positions), 0 refusals. Measured, `python -m ceqjepa.chess_steps`:

    ARM                             Spearman rho vs exact plies-to-loss, n = 1636
    curv        exp(-kappa)         +0.4122   CI [+0.3686, +0.4540]
    curv_inv    exp(+kappa)         +0.4219   CI [+0.3788, +0.4633]
    hop         unweighted graph    +0.4123   CI [+0.3687, +0.4540]
    euclid      Euclidean control   +0.5105   CI [+0.4715, +0.5475]
    move_graph  exact move graph    +0.7075   CI [+0.6809, +0.7322]

Read it in that order. Curvature reweighting is worth -0.0001 rho against the
IDENTICAL unweighted edge set -- not a tie inside a wide interval, a tie in the
fourth decimal place. Inverting the orientation moves it by 0.0097, an order of
magnitude inside the interval, so the direction of the affinity-to-length map is
not load-bearing HERE, and that is a symptom of the same thing: a weighting that
changes nothing cannot be changed by flipping its sign. And the graph family
LOSES to the straight-line control it was supposed to beat, by 0.0983 rho with
non-overlapping intervals -- discretising the encoding into a kNN graph destroys
signal the raw distance already had, and curvature does not put it back.

The owner's pre-registered counter was that the arms would TIE because the
encoding already carries plies-to-loss. That was the optimistic reading. The
treatment is not level with its control, it is below it.

WHY, AND IT IS SHARPER THAN "NO EFFECT". The curv and hop geodesics do NOT give
the same ranking. They agree at Spearman +0.9343 over the same 1,636 nodes, so
the reweighting moves 12.7% of the rank variance (1 - 0.9343^2). It moves
it, and the movement carries exactly nothing: +0.4122 against +0.4123 on the
target, identical to the fourth decimal. Curvature reweighting on this graph is
not inert, it is NOISE -- it reorders nodes, and the reordering is orthogonal to
plies-to-loss. That is worse than a tie, because a tie is at least consistent
with "the signal was already there".

The scale of it is not the problem. kappa here is real and well spread -- mean
-0.1270, CI [-0.1312, -0.1228], min -0.7917, max +0.5625 over 10,125 edges -- so
edge lengths exp(-kappa) span [0.5698, 2.2071], a factor of 3.9. A perturbation
that large reaches deep enough to reorder, and still lands nowhere useful. Nor
is the direction the problem: flipping the map to exp(+kappa) moves rho by
+0.0097 against a CI half-width of 0.0427, so the orientation the coordinator
flagged is correctly stated here and simply has nothing to act on.

AND IT IS NOT A RESOLUTION LIMIT -- the arm that WINS has less of it. The kNN
geodesic runs 2.6 edges to the basin and separates the 9 label levels present
into 5 distinct distances. move_graph, on the SAME 364 anchors and the SAME
1,636 scored nodes, separates them into 3, and reaches +0.7075 against +0.4122.
Three buckets beat five by +0.2953 rho because they are the right three. Nothing
about the losing arms is repaired by giving them a finer graph: they are cutting
the space along the wrong seams, and a finer wrong seam is still wrong.

THE SURVIVING ROUTE, and it is in the table. move_graph at +0.7075 is the same
kind of object -- a graph geodesic to the same basin, scored on the same nodes
against the same labels -- and it beats the Euclidean control by +0.1970 with
non-overlapping intervals. The only difference is WHICH EDGES EXIST: its edges
are the legal moves, not the 8 nearest coordinate neighbours. The lever on this
bed is the edge set, not the edge weight, and a reweighting scheme can only
redistribute value inside an edge set that already holds it.

PRICED BY A RUN, AND PUBLISHED AS A SPAN. The next experiment is Ollivier
curvature on the move graph itself. move_graph_price() builds the induced
subgraph on the 3,000 move-graph nodes nearest the mate set, runs
curvature_from_graph over it, and times that call with perf_counter in the
running process; demo() (j) prints what it measured on the box the run happened
on. The ball's STRUCTURAL facts are deterministic and reproduce exactly
everywhere: 5,534 edges, 0 refusals, kappa mean -0.1103 sd 0.1904 in [-0.7807,
+0.5000], ball mean degree 3.7 against the full graph's 26.6.

THE RATE IS NOT DETERMINISTIC AND IS NOT PUBLISHED AS IF IT WERE. Runs of this
same function have returned rates from 1.678 ms/edge to 4.116 ms/edge across two
machines and several loads -- three of those breaches happened in one afternoon
on ONE machine as other work finished around it -- so the extrapolation over
4,891,672 undirected edges is a RANGE, 2.3 h to 5.6 h single-core, and
PRICE_OBSERVED_MS_PER_EDGE carries both ends. It is a lower
bound on a lower bound: lower once because the rate comes from ball degree 3.7,
where each transport program is about 22 cells, while the full graph runs at
degree 26.6 and about 759; and lower again because the slowest machine that
exists is not among the two that were timed.

WHY THE UNDIRECTED EDGE COUNT EQUALS THE DIRECTED ARC COUNT. The move relation is
bipartite by side to move: every arc runs between a White-to-move position and a
Black-to-move one, and an arc's reverse would have to be a move by the side that
did not just move. So no arc has a reciprocal, symmetrising doubles the nonzeros
exactly, and W.nnz // 2 is 4,891,672 -- the directed arc count itself. That
sentence was true and bound to nothing for a full round; move_graph_price now
COUNTS the reciprocal pairs and demo() (j) prints the count, with
test_the_move_graph_has_no_reciprocal_arcs recomputing it from the raw arc list.

RETIRED, KEPT ON THE PAGE, AND THE CHARGE AGAINST IT IS NARROWER THAN IT WAS. An
earlier revision stated "4.133 ms/edge ... 22.9 s ... >= 20,250 s = >= 5.6 hours"
as measured fact, from a scratch script that never entered the tree, while this
module imported only curvature_from_points and could not compute Ollivier
curvature on a move graph at all. Struck for being UNREPRODUCIBLE, and correctly:
no reader running the module could reach a single one of those figures and no
test pinned any of them. NOT struck for being wrong. Running move_graph_price on
a second box returned 4.116 ms/edge, within 0.4% of the retired 4.133, so the
retired figure reproduces; a revision of this paragraph convicted it on the
strength of one wall clock measuring another wall clock, and that charge is
withdrawn. An unmeasured number is not thereby a false one, and saying so was the
same error in the other direction.

THE GUARD, AND THE LIMIT OF THE GUARD. tests/curvature/test_chess_steps.py
carries test_every_number_in_every_docstring_is_printed_by_the_demo, matching any
plain decimal, any exponent form and any comma-grouped integer, with NO exemption
list, over EVERY docstring this module owns -- selected by __module__ rather than
__all__, so demo() and the private helpers are in scope. Two things it has caught
here. Against the struck revision, scanning the module docstring alone, it named
9 numbers no run printed. Widened to every docstring it named one more, the 1.06
of the Bonett-Wright rank standard error in _corr, which had sat one line below a
function docstring outside the reach of a check that never looked there.

And it would still NOT have caught the strike on its own: 4.133 and 20,250 passed
it, because demo() printed them out of a hardcoded string. Being printed and
being measured are different properties and only the second is worth anything.
What closes that gap is test_the_reroute_price_is_measured_not_asserted, pinning
the published rate to seconds/edges from the same timing -- an identity a literal
cannot satisfy, and one that has been shown to fail when a literal is substituted.

RUN: python -m ceqjepa.chess_steps
"""

import time
from functools import lru_cache

import chess
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import dijkstra
from scipy.stats import pearsonr, spearmanr

from ceqjepa.curvature import (ALPHA, K_NEIGHBORS, Refusal, curvature_from_graph,
                               curvature_from_points, group_stats, is_refusal)

__all__ = [
    "ENDGAME", "SEED", "K_NEIGHBORS", "ALPHA", "N_ANCHOR_MAX", "N_SCORED",
    "EDGE_LENGTH", "EDGE_LENGTH_INVERTED", "ARMS", "KQK_MAX_DTM_PLIES",
    "Refusal", "is_refusal",
    "space", "oracle", "dtm", "committor", "embed",
    "cross_check_forward_search", "steps_report", "one_basin_check",
    "move_graph_price", "BALL_NODES", "demo",
]

#: The enumerated endgame. King and queen against a bare king.
ENDGAME = "KQK"

#: Every sample and every shuffle in this module. PINNED.
SEED = 0

#: The published maximum distance to mate for this endgame, counted in plies from
#: the position the winning side faces. External to this code; the induction has
#: to land on it.
KQK_MAX_DTM_PLIES = 19

#: How many positions are SCORED in the correlation. The basin anchors are in the
#: graph but not scored -- a node at distance 0 with label 0 inflates every arm
#: equally and tells the reader nothing.
N_SCORED = 1636

#: Upper bound on basin anchors placed in the graph. All 364 mates fit under it.
N_ANCHOR_MAX = 400

#: The map from curvature to edge length, and its inverse orientation. Both are
#: strictly positive for every finite kappa, so neither can make a negative cycle.
EDGE_LENGTH = "exp(-kappa)"
EDGE_LENGTH_INVERTED = "exp(+kappa)"

#: Every arm reported, always together. The first three are graph geodesics on
#: one shared edge set; euclid is the straight-line control in the same
#: embedding; move_graph is the exact legal-move graph and is the ceiling.
ARMS = ("curv", "curv_inv", "hop", "euclid", "move_graph")

#: Convergence bar for the committor fixpoint.
Q_TOL = 1e-12

#: Nodes in the move-graph ball whose curvature is actually COMPUTED, so the
#: reroute is priced from a rate this module measured rather than from a literal.
#: PINNED. Chosen as the largest ball that leaves the self-check inside 300 s.
BALL_NODES = 3000

#: Every per-edge rate move_graph_price has actually returned, in ms, as a span.
#: Seven observations: 1.678, 1.965, 2.009, 2.155, 2.199 and 2.785 on the box that
#: wrote this module, at loads ranging from six agents running to none, and 4.116
#: on the Inspector's. A wall clock is a property of the machine and its load, not
#: of this endgame, which is why the price is published as this span and never as
#: whichever end the local run happened to produce. WIDEN IT when an honest run
#: lands outside; that has already happened three times in one afternoon on ONE
#: machine, which is the measurement of how little a single timing is worth.
#:
#: It is documentation of evidence, not a check. Discrimination comes from
#: PRICE_BAND_FACTOR's loose sanity bound plus two things a constant cannot do:
#: match the raw perf_counter reads it is supposed to be the difference of, and
#: come out different on the next run.
PRICE_OBSERVED_MS_PER_EDGE = (1.678, 4.116)

#: How far outside the recorded span a run may land before the sanity bound
#: fires. Deliberately loose: tightening it would only convert honest speed into
#: red tests, and it is not what catches a fabricated clock.
PRICE_BAND_FACTOR = 3.0

#: A second, smaller ball timed separately in the SAME run. Two wall clocks in
#: one process never return the same value, so one fabricated constant cannot be
#: both of them -- and their two independent per-edge rates have to agree about
#: what machine this is. PINNED.
PROBE_NODES = 600

# The refusal codes. Each names a DISTINCT cause, so a caller can count by cause.
STALEMATE = "stalemate"
DRAWN_BY_DEFENCE = "drawn-by-defence"
NOT_IN_THE_SPACE = "not-in-the-space"
CONSTANT_LABEL = "constant-label"
TOO_FEW_POINTS = "too-few-points"

_WK = chess.Piece(chess.KING, chess.WHITE)
_WQ = chess.Piece(chess.QUEEN, chess.WHITE)
_BK = chess.Piece(chess.KING, chess.BLACK)


# ---------------------------------------------------------------------------
# THE SPACE: every legal K+Q vs K position, and every legal move between them
# ---------------------------------------------------------------------------

def _board(wk, wq, bk, white_to_move):
    b = chess.Board(None)
    b.set_piece_at(wk, _WK)
    b.set_piece_at(wq, _WQ)
    b.set_piece_at(bk, _BK)
    b.turn = white_to_move
    return b


@lru_cache(maxsize=1)
def space():
    """The whole legal position space, and the successor CSR over it.

    CLOSED BY CONSTRUCTION: every successor of an enumerated position is itself
    enumerated, with one extra state for the position the queen's capture leads
    to (K vs K, insufficient material, absorbing). A space that is not closed is
    a search with a silent depth cap wearing a different name, which is exactly
    the failure this module exists to avoid.

    Legality is python-chess's own `is_valid()`, and the moves are python-chess's
    own `legal_moves`. Nothing about chess rules is reimplemented here.
    """
    from array import array

    keys, index = [], {}
    for wk in range(64):
        for bk in range(64):
            if bk == wk or chess.square_distance(wk, bk) <= 1:
                continue
            for wq in range(64):
                if wq == wk or wq == bk:
                    continue
                for turn in (chess.WHITE, chess.BLACK):
                    if _board(wk, wq, bk, turn).is_valid():
                        index[(wk, wq, bk, turn)] = len(keys)
                        keys.append((wk, wq, bk, turn))
    n = len(keys)
    sink = n  # queen captured -> K vs K -> absorbing draw

    src, dst = array("i"), array("i")
    indptr = np.zeros(n + 1, np.int64)
    no_move = np.zeros(n, bool)
    in_check = np.zeros(n, bool)
    for i, (wk, wq, bk, turn) in enumerate(keys):
        b = _board(wk, wq, bk, turn)
        count = 0
        for m in b.legal_moves:
            f, t = m.from_square, m.to_square
            if turn == chess.WHITE:
                j = index[(t, wq, bk, chess.BLACK)] if f == wk \
                    else index[(wk, t, bk, chess.BLACK)]
            else:
                j = sink if t == wq else index[(wk, wq, t, chess.WHITE)]
            src.append(i)
            dst.append(j)
            count += 1
        indptr[i + 1] = count
        if count == 0:
            no_move[i] = True
            in_check[i] = b.is_check()
    indptr = np.cumsum(indptr)

    return dict(
        keys=np.array(keys, np.int16), index=index,
        n_positions=n, n_states=n + 1, sink=sink,
        pred=np.frombuffer(src, np.int32).copy(),
        succ=np.frombuffer(dst, np.int32).copy(),
        succ_indptr=indptr, n_edges=len(src),
        no_move=no_move, in_check=in_check,
    )


def _reverse_csr(sp):
    """Predecessor lists, as (indptr, flat). Built once, used by the induction."""
    nt = sp["n_states"]
    order = np.argsort(sp["succ"], kind="stable")
    ptr = np.zeros(nt + 1, np.int64)
    ptr[1:] = np.cumsum(np.bincount(sp["succ"], minlength=nt))
    return ptr, sp["pred"][order]


def _gather(ptr, flat, nodes):
    """The concatenated predecessor lists of `nodes`, vectorised."""
    lo, hi = ptr[nodes], ptr[nodes + 1]
    counts = hi - lo
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64)
    offs = np.zeros(counts.size + 1, np.int64)
    offs[1:] = np.cumsum(counts)
    idx = np.repeat(lo, counts) + (np.arange(total) - np.repeat(offs[:-1], counts))
    return flat[idx].astype(np.int64)


UNLABELLED, WIN, LOSS, DRAW = 0, 1, 2, 3


@lru_cache(maxsize=1)
def oracle():
    """Exact distance-to-mate for every position, plus the exact committor.

    The induction runs to its FIXPOINT. There is no ply limit anywhere in it: it
    stops when a layer labels nothing new, and every position it never labels is
    a draw, reported as such and counted by cause.
    """
    sp = space()
    n, nt, sink = sp["n_positions"], sp["n_states"], sp["sink"]
    ptr, flat = _reverse_csr(sp)
    out_degree = np.zeros(nt, np.int64)
    out_degree[:n] = np.diff(sp["succ_indptr"])

    label = np.zeros(nt, np.int8)
    plies = np.full(nt, -1, np.int32)
    remaining = out_degree.copy()

    mates = np.flatnonzero(sp["no_move"] & sp["in_check"]).astype(np.int64)
    stalemates = np.flatnonzero(sp["no_move"] & ~sp["in_check"]).astype(np.int64)
    label[mates] = LOSS
    plies[mates] = 0
    label[stalemates] = DRAW
    label[sink] = DRAW

    frontier, depth, layers = mates, 0, 0
    while frontier.size:
        won = np.unique(_gather(ptr, flat, frontier))
        won = won[label[won] == UNLABELLED]
        if won.size == 0:
            break
        label[won] = WIN
        plies[won] = depth + 1
        preds = _gather(ptr, flat, won)
        remaining -= np.bincount(preds, minlength=nt)
        lost = np.unique(preds[(remaining[preds] <= 0) & (label[preds] == UNLABELLED)])
        label[lost] = LOSS
        plies[lost] = depth + 2
        frontier, depth, layers = lost, depth + 2, layers + 1

    # every position the fixpoint never labelled is a draw held by the defence
    held = np.flatnonzero(label[:n] == UNLABELLED)
    label[held] = DRAW

    q, residual, iters = _committor(sp, mates, stalemates)

    white = sp["keys"][:, 3].astype(bool)
    win_basin = plies[:n] >= 0
    return dict(
        dtm=plies[:n], label=label[:n], q=q[:n],
        n_win=int(win_basin.sum()),
        n_draw=int(n - win_basin.sum()),
        n_unresolved=0,
        layers=layers,
        max_dtm_white_to_move=int(plies[:n][white & win_basin].max()),
        max_dtm_black_to_move=int(plies[:n][~white & win_basin].max()),
        mate_nodes=mates, stalemate_nodes=stalemates,
        draw_absorbing_nodes=stalemates,
        held_draw_nodes=held,
        refusals={STALEMATE: int(stalemates.size), DRAWN_BY_DEFENCE: int(held.size)},
        q_residual=residual, q_iterations=iters,
    )


def _committor(sp, mates, stalemates):
    """P(reach mate before a draw) under uniform random play, by linear fixpoint.

    Not a simulation and not a truncation: the iteration runs until the update is
    below Q_TOL and the returned residual ||q - Pq||_inf is measured afterwards
    on the fixed point, so the number quoted is the one that was reached.
    """
    nt = sp["n_states"]
    degree = np.zeros(nt, np.int64)
    degree[:sp["n_positions"]] = np.diff(sp["succ_indptr"])
    weight = 1.0 / degree[sp["pred"]]
    P = sparse.csr_matrix((weight, (sp["pred"], sp["succ"])), shape=(nt, nt))

    boundary = np.zeros(nt, bool)
    boundary[mates] = boundary[stalemates] = True
    boundary[sp["sink"]] = True
    fixed = np.zeros(nt)
    fixed[mates] = 1.0

    q = fixed.copy()
    for iters in range(1, 100001):
        nxt = P @ q
        nxt[boundary] = fixed[boundary]
        step = float(np.abs(nxt - q).max())
        q = nxt
        if step < Q_TOL:
            break
    q[mates] = 1.0
    q[stalemates] = 0.0
    q[sp["sink"]] = 0.0
    res = np.abs(P @ q - q)
    res[boundary] = 0.0
    return q, float(res.max()), iters


# ---------------------------------------------------------------------------
# LOOKING ONE POSITION UP
# ---------------------------------------------------------------------------

def _key_of(fen):
    b = chess.Board(fen)
    wk = b.king(chess.WHITE)
    bk = b.king(chess.BLACK)
    queens = list(b.pieces(chess.QUEEN, chess.WHITE))
    if (wk is None or bk is None or len(queens) != 1
            or len(b.piece_map()) != 3 or not b.is_valid()):
        return None
    return (wk, queens[0], bk, b.turn)


def dtm(fen):
    """Exact plies to mate for `fen`, or a Refusal saying why there is no number.

    A drawn position is NOT a mate in zero. It comes back as a value in the
    output space carrying its cause, so a caller can count draws by kind and a
    threshold applied to one raises instead of quietly passing.
    """
    key = _key_of(fen)
    if key is None:
        return Refusal(NOT_IN_THE_SPACE,
                       "%r is not a legal %s position: this oracle is exact over "
                       "that endgame and has nothing to say about any other"
                       % (fen, ENDGAME))
    sp, o = space(), oracle()
    i = sp["index"][key]
    value = int(o["dtm"][i])
    if value >= 0:
        return value
    if i in set(o["stalemate_nodes"].tolist()):
        return Refusal(STALEMATE,
                       "%r is stalemate: the side to move has no legal move and is "
                       "not in check, so the game ends drawn and there is no "
                       "distance to a mate that never happens" % fen)
    return Refusal(DRAWN_BY_DEFENCE,
                   "%r is drawn with best play: the defence holds, by winning the "
                   "queen or by reaching stalemate, and no forced mate exists to "
                   "measure a distance to" % fen)


def committor(fen):
    """q_loss(fen) under uniform random play, exact to the reported residual."""
    key = _key_of(fen)
    if key is None:
        return Refusal(NOT_IN_THE_SPACE, "%r is not a legal %s position" % (fen, ENDGAME))
    return float(oracle()["q"][space()["index"][key]])


# ---------------------------------------------------------------------------
# THE EMBEDDING
# ---------------------------------------------------------------------------

def embed(idx):
    """6-D encoding of the positions at `idx`: file and rank of each piece, in [0, 1].

    NEUTRAL BY CONSTRUCTION, and that is the whole point of it. Every feature is
    a raw board coordinate. Nothing here is a function of the label -- not the
    exact distance to mate, not the basin probability, and above all not a
    hand-cut chess feature like "how near the defending king is to a corner". A
    feature of that kind would hand the Euclidean control the answer, and the
    comparison below would then be measuring the encoder rather than the graph.

    Squares are distinct triples, so no two rows coincide and the graph metric
    never sees a zero-length edge.
    """
    sq = space()["keys"][np.asarray(idx), :3].astype(float)
    files = sq % 8.0
    ranks = sq // 8.0
    return np.column_stack([files[:, 0], ranks[:, 0], files[:, 1], ranks[:, 1],
                            files[:, 2], ranks[:, 2]]) / 7.0


# ---------------------------------------------------------------------------
# THE SECOND ALGORITHM
# ---------------------------------------------------------------------------

def cross_check_forward_search(max_plies=5, n_sample=300, seed=SEED):
    """Forward minimax, sharing no code with the backward induction.

    Recursive negamax over python-chess's own move generation, iterative
    deepening to `max_plies`, returning the exact ply count when a forced mate is
    found inside that horizon and None otherwise. Compared against the induction
    on a random sample of positions whose true distance is at most `max_plies`.
    One number reached by two algorithms is the only check that catches an
    indexing bug in either; a single algorithm agreeing with itself catches
    nothing.
    """
    sp, o = space(), oracle()
    plies = o["dtm"]
    pool = np.flatnonzero((plies >= 0) & (plies <= max_plies))
    rng = np.random.default_rng(seed)
    pick = rng.choice(pool, size=min(n_sample, pool.size), replace=False)

    def mate_in(board, limit):
        """Exact plies to mate from `board` if <= limit, else None."""
        if board.is_checkmate():
            return 0
        if limit == 0 or not board.legal_moves:
            return None
        best = None
        maximise = board.turn == chess.BLACK  # the defender stretches it out
        for m in board.legal_moves:
            board.push(m)
            sub = mate_in(board, limit - 1)
            board.pop()
            if sub is None:
                if maximise:
                    return None  # one escape is enough for the defence
                continue
            cand = sub + 1
            if best is None or (cand > best if maximise else cand < best):
                best = cand
        return best

    worst = 0
    for i in pick:
        wk, wq, bk, turn = (int(v) for v in sp["keys"][i])
        got = mate_in(_board(wk, wq, bk, bool(turn)), max_plies)
        if got != int(plies[i]):
            worst += 1
    return worst, int(pick.size)


# ---------------------------------------------------------------------------
# T-STEPS
# ---------------------------------------------------------------------------

def _corr(distance, label, z=1.96):
    """Spearman and Pearson with CIs, or a Refusal saying why there is neither.

    Spearman is the headline because curvature is invariant under scaling the
    edge weights, so raw path lengths are not comparable across arms and only the
    ORDER they induce is. The CI is Fisher-z with the Bonett-Wright standard
    error 1.06/sqrt(n-3) for the rank statistic.
    """
    d = np.asarray(distance, float)
    y = np.asarray(label, float)
    if d.size < 5:
        return Refusal(TOO_FEW_POINTS,
                       "%d points is not enough to put a confidence interval on a "
                       "correlation" % d.size)
    if d.std() == 0.0 or y.std() == 0.0:
        return Refusal(CONSTANT_LABEL,
                       "the labels (or the distances) have zero variance across "
                       "these %d points -- drawn from a single distance level, "
                       "there is no relationship to correlate and a number here "
                       "would be an artefact of the tie-breaking, not a finding"
                       % d.size)
    rho = float(spearmanr(d, y).statistic)
    r = float(pearsonr(d, y).statistic)
    se = 1.06 / np.sqrt(d.size - 3)
    lo, hi = np.tanh(np.arctanh(rho) - z * se), np.tanh(np.arctanh(rho) + z * se)
    se_p = 1.0 / np.sqrt(d.size - 3)
    return dict(rho=rho, rho_ci_lo=float(lo), rho_ci_hi=float(hi),
                pearson=r,
                pearson_ci_lo=float(np.tanh(np.arctanh(r) - z * se_p)),
                pearson_ci_hi=float(np.tanh(np.arctanh(r) + z * se_p)),
                n=int(d.size))


def _basin_distance(n_nodes, edges, lengths, anchors):
    """Multi-source shortest path to the anchor set, on ONE shared edge set."""
    if not edges:
        return np.full(n_nodes, np.inf)
    ii = np.array([e[0] for e in edges])
    jj = np.array([e[1] for e in edges])
    W = sparse.csr_matrix((lengths, (ii, jj)), shape=(n_nodes, n_nodes))
    W = W + W.T
    return dijkstra(W, directed=False, indices=anchors, min_only=True)


@lru_cache(maxsize=4)
def _bed(n_scored=N_SCORED, seed=SEED, level=None):
    """One embedding, one kNN graph, one edge set, shared by every graph arm.

    `level` restricts the SCORED positions to a single distance level, which is
    the one-basin planted negative: the anchors are still there, the geometry is
    still built, and the label simply has no variance.
    """
    sp, o = space(), oracle()
    plies = o["dtm"]
    black = ~sp["keys"][:, 3].astype(bool)
    anchors = np.flatnonzero((plies == 0) & black)[:N_ANCHOR_MAX]
    pool = np.flatnonzero((plies > 0) & black) if level is None \
        else np.flatnonzero((plies == level) & black)
    rng = np.random.default_rng(seed)
    scored = rng.choice(pool, size=min(n_scored, pool.size), replace=False)

    nodes = np.concatenate([anchors, scored])
    X = embed(nodes)
    res = curvature_from_points(X, k=K_NEIGHBORS, alpha=ALPHA)
    kappa = np.asarray(res["kappa"], float)
    return dict(sp=sp, plies=plies, nodes=nodes, X=X, res=res, kappa=kappa,
                n_anchor=int(anchors.size), n_scored=int(scored.size),
                anchor_slice=np.arange(anchors.size),
                scored_slice=np.arange(anchors.size, nodes.size))


@lru_cache(maxsize=1)
def _move_graph():
    """The exact legal-move relation over the whole space, symmetrised."""
    sp = space()
    W = sparse.csr_matrix((np.ones(sp["n_edges"]), (sp["pred"], sp["succ"])),
                          shape=(sp["n_states"], sp["n_states"]))
    return ((W + W.T) > 0).astype(float).tocsr()


@lru_cache(maxsize=1)
def _move_graph_distance():
    """Hop distance to the nearest mate on the EXACT legal-move graph.

    The ceiling: the best any graph geodesic could report, since this graph IS
    the move relation the plies are counted in. It is not the same number as
    plies-to-loss -- it ignores the adversary, so it is the COOPERATIVE mate
    distance -- and the gap between the two is what makes the correlation hard
    rather than trivial.
    """
    return dijkstra(_move_graph(), directed=False, indices=oracle()["mate_nodes"],
                    min_only=True)


@lru_cache(maxsize=1)
def move_graph_price(n_ball=BALL_NODES):
    """MEASURE what the reroute costs, in this process, and never assert it.

    THIS FUNCTION EXISTS BECAUSE OF A STRIKE. An earlier revision priced the
    reroute at "4.133 ms/edge over a 3000-node ball, 22.9 s, >= 5.6 hours" and
    demo() printed all of it as fact. The measurement was real, but it was made
    in a scratch script that never entered the tree: the module imported only
    curvature_from_points and could not compute Ollivier curvature on a move
    graph at all, so no reader running it could reproduce a single one of those
    figures and no test pinned them. Consistent arithmetic on an unmeasured rate
    is indistinguishable from consistent arithmetic on a measured one, which is
    exactly why it has to be measured HERE.

    So: build the induced subgraph on the n_ball positions nearest the mate set
    in the move graph, run curvature_from_graph over it, time that call with
    perf_counter, and derive every downstream number from the timing. The
    extrapolation to the full graph is a LOWER bound and carries its reason --
    the rate is measured at the ball's degree, and the full graph runs at a much
    higher one, where each transport program has many more cells.
    """
    sp, W = space(), _move_graph()

    # WHY W.nnz // 2 IS THE DIRECTED ARC COUNT, counted rather than asserted. The
    # move relation is bipartite by side to move: every arc runs between a
    # White-to-move position and a Black-to-move one, and an arc's reverse would
    # have to be a move by the side that did not just move. So no arc has a
    # reciprocal and symmetrising doubles the nonzeros exactly. This was prose
    # before it was executable, which is the same defect class as an unmeasured
    # price: correct arithmetic resting on nothing that runs.
    nstate = sp["n_states"]
    arc = np.sort(sp["pred"].astype(np.int64) * nstate + sp["succ"])
    back = sp["succ"].astype(np.int64) * nstate + sp["pred"]
    at = np.clip(np.searchsorted(arc, back), 0, arc.size - 1)
    n_reciprocal = int((arc[at] == back).sum())

    order = np.argsort(_move_graph_distance()[:sp["n_positions"]],
                       kind="stable")[:n_ball]
    sub = np.asarray(W[order][:, order].todense())

    # TWO CLOCKS, and the reason there are two. Everything this function returns
    # is homogeneous of degree one in `seconds`: replace that one root with a
    # literal and ms_per_edge, lower_seconds and lower_hours all follow it
    # exactly, every internal identity still holds, and the fabrication is
    # invisible to any test that only checks those identities. That is the defect
    # this module's own docstring named and then did not close. A wall clock
    # never returns the same value twice, so the root is timed twice over
    # different work, the two rates must agree about the machine, and the
    # published rate must land inside the span recorded across earlier runs.
    #
    # The exact bind is the pair of RAW perf_counter reads each interval is the
    # difference of. Those are returned, so `seconds` is not merely asserted to be
    # a duration -- it has to equal t_ball1 - t_ball0 exactly, and the four reads
    # have to be strictly increasing and fit inside the read taken on the way out.
    # A literal at the root fails that identity immediately, and it does so on any
    # machine, which the calibration span cannot claim.
    probe_order = order[:PROBE_NODES]
    probe_sub = np.asarray(W[probe_order][:, probe_order].todense())
    t_probe0 = time.perf_counter()
    probe_res = curvature_from_graph(probe_sub)
    t_probe1 = time.perf_counter()
    probe_seconds = t_probe1 - t_probe0
    probe_attempted = probe_res["n_edges"] + probe_res["n_refused"]

    t_ball0 = time.perf_counter()
    res = curvature_from_graph(sub)
    t_ball1 = time.perf_counter()
    seconds = t_ball1 - t_ball0

    attempted = res["n_edges"] + res["n_refused"]
    ms_per_edge = 1000.0 * seconds / attempted
    probe_ms_per_edge = 1000.0 * probe_seconds / max(probe_attempted, 1)
    rate_ratio = (max(ms_per_edge, probe_ms_per_edge)
                  / min(ms_per_edge, probe_ms_per_edge))
    full_edges = int(W.nnz // 2)
    ball_degree = float(sub.sum() / n_ball)
    full_degree = float(W.nnz / sp["n_states"])
    lower_seconds = ms_per_edge / 1000.0 * full_edges
    lo_ms, hi_ms = PRICE_OBSERVED_MS_PER_EDGE
    t_return = time.perf_counter()
    return dict(
        n_reciprocal_arcs=n_reciprocal,
        t_probe0=float(t_probe0), t_probe1=float(t_probe1),
        t_ball0=float(t_ball0), t_ball1=float(t_ball1), t_return=float(t_return),
        probe_nodes=int(PROBE_NODES), probe_attempted=int(probe_attempted),
        probe_seconds=float(probe_seconds),
        probe_ms_per_edge=float(probe_ms_per_edge),
        rate_ratio=float(rate_ratio),
        rate_in_span=bool(lo_ms <= ms_per_edge <= hi_ms),
        rate_in_band=bool(lo_ms / PRICE_BAND_FACTOR <= ms_per_edge
                          <= hi_ms * PRICE_BAND_FACTOR),
        observed_lo_ms=float(lo_ms), observed_hi_ms=float(hi_ms),
        span_lo_hours=float(lo_ms / 1000.0 * full_edges / 3600.0),
        span_hi_hours=float(hi_ms / 1000.0 * full_edges / 3600.0),
        n_ball=int(n_ball), n_edges=int(res["n_edges"]),
        n_refused=int(res["n_refused"]), refusals=res["refusals"],
        n_attempted=int(attempted), kappa=group_stats(res["kappa"]),
        seconds=float(seconds), ms_per_edge=float(ms_per_edge),
        full_graph_edges=full_edges,
        ball_degree=ball_degree, full_degree=full_degree,
        cells_ball=(1.0 + ball_degree) ** 2, cells_full=(1.0 + full_degree) ** 2,
        lower_seconds=float(lower_seconds),
        lower_hours=float(lower_seconds / 3600.0),
    )


@lru_cache(maxsize=4)
def _arms(n_scored=N_SCORED, seed=SEED, level=None):
    """Every arm's distance vector and the label, on matched nodes."""
    bed = _bed(n_scored, seed, level)
    res, kappa = bed["res"], bed["kappa"]
    n = bed["nodes"].size
    edges = res["edges"]
    anchors = bed["anchor_slice"]

    d_curv = _basin_distance(n, edges, np.exp(-kappa), anchors)
    d_inv = _basin_distance(n, edges, np.exp(+kappa), anchors)
    d_hop = _basin_distance(n, edges, np.ones(kappa.size), anchors)
    d_euc = np.linalg.norm(bed["X"][:, None, :] - bed["X"][None, anchors, :],
                           axis=2).min(axis=1)
    d_move = _move_graph_distance()[bed["nodes"]]

    keep = bed["scored_slice"]
    finite = np.isfinite(d_curv[keep]) & np.isfinite(d_hop[keep]) \
        & np.isfinite(d_inv[keep]) & np.isfinite(d_move[keep])
    keep = keep[finite]
    label = bed["plies"][bed["nodes"][keep]].astype(float)

    # the SAME geodesic routine, run from a node to itself on the same graph
    probe = [int(v) for v in keep[:5]]
    self_geo = max(float(_basin_distance(n, edges, np.exp(-kappa),
                                         np.array([v]))[v]) for v in probe)
    self_euc = float(np.linalg.norm(bed["X"][probe] - bed["X"][probe], axis=1).max())

    return dict(bed=bed, keep=keep, label=label,
                n_unreachable=int((~finite).sum()),
                curv=d_curv[keep], curv_inv=d_inv[keep], hop=d_hop[keep],
                euclid=d_euc[keep], move_graph=d_move[keep],
                self_geodesic_max=self_geo, self_euclid_max=self_euc)


@lru_cache(maxsize=1)
def steps_report(n_scored=N_SCORED, seed=SEED):
    """Every arm, with its control, at matched nodes and matched density."""
    a = _arms(n_scored, seed, None)
    bed = a["bed"]
    res, kappa = bed["res"], bed["kappa"]
    rng = np.random.default_rng(seed)
    shuffled_label = rng.permutation(a["label"])

    out = dict(
        endgame=ENDGAME, seed=seed, k=K_NEIGHBORS, alpha=ALPHA,
        edge_length=EDGE_LENGTH, edge_length_inverted=EDGE_LENGTH_INVERTED,
        n_graph_nodes=int(bed["nodes"].size), n_anchor=bed["n_anchor"],
        n_dims=int(bed["X"].shape[1]),
        n_unreachable=a["n_unreachable"],
        kappa=group_stats(kappa),
        n_refused=res["n_refused"], refusals=res["refusals"],
        min_edge_length=float(np.exp(-kappa).min()),
        max_edge_length=float(np.exp(-kappa).max()),
        min_edge_length_inverted=float(np.exp(+kappa).min()),
        self_geodesic_max=a["self_geodesic_max"],
        self_euclid_max=a["self_euclid_max"],
        shuffled={},
    )
    for arm in ARMS:
        c = _corr(a[arm], a["label"])
        assert not is_refusal(c), (arm, c)
        c["n_graph_edges"] = res["n_edges"]
        c["n_graph_nodes"] = int(bed["nodes"].size)
        out[arm] = c
        sh = _corr(a[arm], shuffled_label)
        assert not is_refusal(sh), (arm, sh)
        out["shuffled"][arm] = sh

    # THE MECHANISM. Reporting that curvature changed nothing is half a result.
    # The other half is showing the two geodesics ARE the same ranking, and how
    # long a path has to be before a multiplicative reweighting washes out of it.
    out["curv_vs_hop_rho"] = float(spearmanr(a["curv"], a["hop"]).statistic)
    out["mean_path_edges"] = float(np.mean(a["hop"]))
    # the ceiling arm's path length, so "the kNN paths were too short" can be
    # ruled out as the explanation rather than left as an excuse
    out["mean_move_path_edges"] = float(np.mean(a["move_graph"]))
    out["n_label_levels"] = int(np.unique(a["label"]).size)
    out["n_hop_levels"] = int(np.unique(a["hop"]).size)
    out["n_move_levels"] = int(np.unique(a["move_graph"]).size)
    out["curv_len_ratio"] = float(np.exp(-kappa).max() / np.exp(-kappa).min())

    out["winner"] = max(ARMS, key=lambda arm: out[arm]["rho"])
    out["curv_minus_euclid"] = out["curv"]["rho"] - out["euclid"]["rho"]
    out["curv_minus_hop"] = out["curv"]["rho"] - out["hop"]["rho"]
    out["curv_minus_inverted"] = out["curv"]["rho"] - out["curv_inv"]["rho"]
    beaten = out["euclid"]["rho"] > out["curv"]["rho_ci_hi"]
    out["verdict"] = (
        "curvature reweighting is worth %+.4f rho against the IDENTICAL unweighted "
        "edge set, while moving %.1f%% of the rank variance -- it reorders, and the "
        "reordering is orthogonal to the target, so it is noise and not a tie. "
        "The Euclidean control %s the curvature arm (%+.4f vs %+.4f, %s intervals). "
        "Best arm %r at %+.4f, differing from the rest only in WHICH EDGES EXIST."
        % (out["curv_minus_hop"],
           100.0 * (1.0 - out["curv_vs_hop_rho"] ** 2),
           "BEATS" if beaten else "does not beat",
           out["euclid"]["rho"], out["curv"]["rho"],
           "non-overlapping" if beaten else "overlapping",
           out["winner"], out[out["winner"]]["rho"]))
    price = move_graph_price()
    out["price"] = price
    out["reroute"] = (
        "the lever is the edge set, not the edge weight. Ollivier curvature ON THE "
        "MOVE GRAPH, priced from %.3f ms/edge MEASURED in this process over a "
        "%d-node induced ball (%d edges in %.1f s) against %d undirected edges. "
        "PUBLISHED AS A SPAN, because a rate is a wall clock: two runs of this "
        "function on two machines gave %.3f and %.3f ms/edge, so the full graph is "
        "%.1f h to %.1f h single-core. It is a lower bound on a lower bound -- "
        "lower once because the rate comes from ball degree %.1f (about %.0f "
        "transport cells) while the full graph runs at degree %.1f (about %.0f), "
        "and lower again because the slowest machine that exists is not in the "
        "span. Not blocked, just not 300 s."
        % (price["ms_per_edge"], price["n_ball"], price["n_attempted"],
           price["seconds"], price["full_graph_edges"],
           price["observed_lo_ms"], price["observed_hi_ms"],
           price["span_lo_hours"], price["span_hi_hours"],
           price["ball_degree"], price["cells_ball"],
           price["full_degree"], price["cells_full"]))
    return out


@lru_cache(maxsize=1)
def one_basin_check(level=8, n_scored=400, seed=SEED):
    """THE PLANTED NEGATIVE: every scored position drawn from ONE distance level.

    The full pipeline runs -- same encoder, same kNN graph, same anchors, same
    geodesics -- and the only thing missing is variance in the label. Every arm
    must come back a Refusal. An arm that returns any number here is reading its
    own tie-breaking as a result.
    """
    a = _arms(n_scored, seed, level)
    out = {arm: _corr(a[arm], a["label"]) for arm in ARMS}
    out["level"] = level
    out["n"] = int(a["label"].size)
    return out


# ---------------------------------------------------------------------------
# SELF-CHECK
# ---------------------------------------------------------------------------

def _line(name, c):
    return ("    %-11s rho %+.4f  CI [%+.4f, %+.4f]   pearson %+.4f   n %d"
            % (name, c["rho"], c["rho_ci_lo"], c["rho_ci_hi"], c["pearson"], c["n"]))


def demo():
    t0 = time.time()

    print("(a) THE SPACE. python-chess enumerates it; nothing about the rules is")
    print("    reimplemented here. Closed under moves, plus one absorbing state for")
    print("    the position the queen's capture leads to.")
    sp = space()
    print("    %s: %d legal positions, %d legal moves, %d states (+1 for K vs K)"
          % (ENDGAME, sp["n_positions"], sp["n_edges"], sp["n_states"]))
    assert int(sp["succ"].max()) < sp["n_states"]
    assert int(sp["succ_indptr"][-1]) == sp["n_edges"]
    print("    closed: max successor index %d < %d states, %.1f s"
          % (sp["succ"].max(), sp["n_states"], time.time() - t0))

    print("(b) THE EXACT ORACLE, by backward induction to the FIXPOINT. No ply")
    print("    limit: the loop stops when a layer labels nothing new.")
    o = oracle()
    print("    layers %d   win basin %d   draws %d   unresolved %d"
          % (o["layers"], o["n_win"], o["n_draw"], o["n_unresolved"]))
    print("    max distance-to-mate  White to move %d plies   Black to move %d plies"
          % (o["max_dtm_white_to_move"], o["max_dtm_black_to_move"]))
    assert o["n_unresolved"] == 0
    assert o["n_win"] + o["n_draw"] == sp["n_positions"]
    assert o["max_dtm_white_to_move"] == KQK_MAX_DTM_PLIES, o["max_dtm_white_to_move"]
    print("    EXTERNAL CHECK: %d plies is the published maximum for %s (mate in"
          % (KQK_MAX_DTM_PLIES, ENDGAME))
    print("    10 moves). This code did not produce that number and lands on it.")
    known = [("4k3/4Q3/4K3/8/8/8/8/8 b - - 0 1", 0, "mate on the board"),
             ("4k3/8/4K3/4Q3/8/8/8/8 w - - 0 1", 1, "Qe5-h8# or Qe5-b8#"),
             ("8/8/8/8/8/8/8/k1KQ4 b - - 0 1", 2, "every reply allows mate")]
    for fen, want, why in known:
        got = dtm(fen)
        print("    %-34s -> %d plies  (%s)" % (fen.split(" ")[0], got, why))
        assert got == want, (fen, got, want)

    print("(c) THE SECOND ALGORITHM. Forward negamax over python-chess move")
    print("    generation, sharing no code with the induction.")
    bad, n = cross_check_forward_search(max_plies=5, n_sample=300)
    print("    disagreements: %d of %d sampled positions at <= 5 plies" % (bad, n))
    assert bad == 0

    print("(d) DRAWS ARE REFUSALS, NOT ZEROS, and they carry which kind.")
    held = int(o["held_draw_nodes"][0])
    wk, wq, bk, turn = (int(v) for v in sp["keys"][held])
    for fen in ("k7/2Q5/8/8/8/8/8/K7 b - - 0 1", _board(wk, wq, bk, bool(turn)).fen()):
        v = dtm(fen)
        print("    %-30s %s: %s" % (fen.split(" ")[0], v.code, v.reason[:78]))
        assert is_refusal(v)
    print("    census over the whole space: %r" % (o["refusals"],))
    assert len(o["refusals"]) >= 2
    q = o["q"]
    interior = q[(q > 0.0) & (q < 1.0)]
    print("    COMMITTOR q_loss, uniform random play, %d iterations at Q_TOL = %g:"
          % (o["q_iterations"], Q_TOL))
    print("    ||q - Pq||inf = %.3e   interior values %d   max %.4f   mean %.4f"
          % (o["q_residual"], interior.size, interior.max(), interior.mean()))
    assert o["q_residual"] < 1e-9
    assert float(np.abs(q[o["mate_nodes"]] - 1.0).max()) == 0.0
    assert interior.size > 1000
    print("    Under OPTIMAL play the same committor is 0 or 1 and has no gradient")
    print("    at all: that degeneracy is a property of a deterministic game.")

    print("(e) T-STEPS, WITH ITS CONTROLS. One embedding, one kNN graph at k = %d,"
          % K_NEIGHBORS)
    print("    alpha = %g, one edge set. Only the edge LENGTH changes between the" % ALPHA)
    print("    three graph arms. Rank correlation, because curvature is invariant")
    print("    under scaling the weights and raw path lengths are not comparable.")
    print("    CI: Fisher-z on the rank statistic with the Bonett-Wright standard")
    print("    error 1.06/sqrt(n-3); the Pearson interval uses 1/sqrt(n-3).")
    r = steps_report()
    print("    encoding: %d-D raw board coordinates, no feature of the label in it."
          % r["n_dims"])
    print("    nodes %d (%d basin anchors, not scored)   edges %d   refused %d %r"
          % (r["n_graph_nodes"], r["n_anchor"], r["kappa"]["n_edges"],
             r["n_refused"], r["refusals"]))
    print("    kappa mean %+.4f  CI [%+.4f, %+.4f]  min %+.4f  max %+.4f  EDGES %d"
          % (r["kappa"]["mean"], r["kappa"]["ci_lo"], r["kappa"]["ci_hi"],
             r["kappa"]["min"], r["kappa"]["max"], r["kappa"]["n_edges"]))
    print("    edge length %s in [%.4f, %.4f]   inverted %s min %.4f   (both > 0,"
          % (EDGE_LENGTH, r["min_edge_length"], r["max_edge_length"],
             EDGE_LENGTH_INVERTED, r["min_edge_length_inverted"]))
    print("    so no edge is a negative cycle)")
    for arm in ARMS:
        print(_line(arm, r[arm]))
    print("    curv - euclid %+.4f    curv - hop %+.4f    best arm %r"
          % (r["curv_minus_euclid"], r["curv_minus_hop"], r["winner"]))
    print("    move_graph - euclid %+.4f: the ceiling beats the control the graph"
          % (r["move_graph"]["rho"] - r["euclid"]["rho"]))
    print("    arms lost to, on the same nodes and the same anchors.")
    print("    VERDICT: %s" % r["verdict"])
    print("    MECHANISM: curv and hop are NOT the same ranking -- they agree at rho")
    print("    %+.4f, so the reweighting moves %.1f%% of the rank variance, over edge"
          % (r["curv_vs_hop_rho"], 100.0 * (1.0 - r["curv_vs_hop_rho"] ** 2)))
    print("    lengths spanning a factor of %.1f. It reorders, and the reordering is"
          % r["curv_len_ratio"])
    print("    orthogonal to the target: %+.4f vs %+.4f. Not inert -- NOISE. Worse"
          % (r["curv"]["rho"], r["hop"]["rho"]))
    print("    than a tie, which would at least be consistent with the signal")
    print("    already being there.")
    print("    NOT A RESOLUTION LIMIT -- the arm that WINS has less of it. kNN paths")
    print("    run %.1f edges and separate %d label levels into %d distances;"
          % (r["mean_path_edges"], r["n_label_levels"], r["n_hop_levels"]))
    print("    move_graph runs %.1f edges and separates them into %d, on the SAME"
          % (r["mean_move_path_edges"], r["n_move_levels"]))
    print("    anchors and the SAME nodes, and reaches %+.4f against %+.4f. %d buckets"
          % (r["move_graph"]["rho"], r["curv"]["rho"], r["n_move_levels"]))
    print("    beat %d by %+.4f rho because they are the right %d. A finer wrong seam"
          % (r["n_hop_levels"], r["move_graph"]["rho"] - r["curv"]["rho"],
             r["n_move_levels"]))
    print("    is still the wrong seam. THE EDGE SET IS THE LEVER, not the weight.")

    print("(f) ORIENTATION. An affinity is high where two nodes are CLOSE; a length")
    print("    is high where they are FAR, so the map has a direction, it is stated,")
    print("    and both orientations run. On THIS bed neither matters:")
    print("    %s rho %+.4f  vs  %s rho %+.4f   difference %+.4f"
          % (EDGE_LENGTH, r["curv"]["rho"], EDGE_LENGTH_INVERTED,
             r["curv_inv"]["rho"], r["curv_minus_inverted"]))
    half = (r["curv"]["rho_ci_hi"] - r["curv"]["rho_ci_lo"]) / 2.0
    print("    the flip moves rho by %+.4f, %s the curv CI half-width %.4f. Same"
          % (r["curv_minus_inverted"],
             "INSIDE" if abs(r["curv_minus_inverted"]) < half else "OUTSIDE", half))
    print("    finding as (e) from the other side: a weighting that changes nothing")
    print("    cannot be changed by flipping its sign. Orientation is load-bearing")
    print("    wherever the weighting is; here the weighting is not.")
    assert r["curv"]["rho"] != r["curv_inv"]["rho"]

    print("(g) PLANTED NEGATIVE 1: shuffled plies. Same distances, permuted labels.")
    for arm in ARMS:
        s = r["shuffled"][arm]
        print("    %-11s rho %+.4f  CI [%+.4f, %+.4f]" % (arm, s["rho"], s["rho_ci_lo"],
                                                          s["rho_ci_hi"]))
        assert abs(s["rho"]) < 0.15, (arm, s["rho"])
        assert s["rho_ci_lo"] < 0.0 < s["rho_ci_hi"], (arm, s)
    print("    FIRED: every arm collapses into a CI straddling zero.")

    print("(h) PLANTED NEGATIVE 2: every scored position from ONE distance level.")
    one = one_basin_check()
    print("    level %d plies, %d positions, same anchors and same graph"
          % (one["level"], one["n"]))
    for arm in ARMS:
        v = one[arm]
        assert is_refusal(v), (arm, v)
        print("    %-11s %s" % (arm, v.code))
    print("    FIRED: no arm returns a number where the label has no variance.")

    print("(i) PLANTED NEGATIVE 3: the geodesic from a position to itself.")
    print("    curvature-weighted %.17g   Euclidean %.17g"
          % (r["self_geodesic_max"], r["self_euclid_max"]))
    assert r["self_geodesic_max"] == 0.0 and r["self_euclid_max"] == 0.0
    print("    FIRED: exactly zero, not nearly zero.")

    print("(j) THE REROUTE, PRICED BY A MEASUREMENT TAKEN IN THIS PROCESS. Ollivier")
    print("    curvature on the MOVE GRAPH, over the induced ball on the %d nodes"
          % BALL_NODES)
    print("    nearest the mate set. curvature_from_graph, timed with perf_counter.")
    pr = r["price"]
    print("    ball: %d nodes, %d edges (%d refused %r), mean degree %.1f"
          % (pr["n_ball"], pr["n_edges"], pr["n_refused"], pr["refusals"],
             pr["ball_degree"]))
    print("    kappa mean %+.4f  sd %.4f  min %+.4f  max %+.4f  EDGES %d"
          % (pr["kappa"]["mean"], pr["kappa"]["sd"], pr["kappa"]["min"],
             pr["kappa"]["max"], pr["kappa"]["n_edges"]))
    print("    MEASURED %.1f s for %d edges = %.3f ms/edge"
          % (pr["seconds"], pr["n_attempted"], pr["ms_per_edge"]))
    print("    SECOND CLOCK, same run, %d-node sub-ball: %.1f s for %d edges ="
          % (pr["probe_nodes"], pr["probe_seconds"], pr["probe_attempted"]))
    print("    %.3f ms/edge. Two wall clocks never coincide, so one literal cannot"
          % pr["probe_ms_per_edge"])
    print("    be both; and they agree on the machine to a factor of %.2f."
          % pr["rate_ratio"])
    assert pr["seconds"] != pr["probe_seconds"], "two timings returned one value"
    assert pr["probe_attempted"] > 0
    ordered = (pr["t_probe0"] < pr["t_probe1"] < pr["t_ball0"] < pr["t_ball1"]
               < pr["t_return"])
    exact = (pr["seconds"] == pr["t_ball1"] - pr["t_ball0"]
             and pr["probe_seconds"] == pr["t_probe1"] - pr["t_probe0"])
    print("    both intervals are differences of RAW perf_counter reads that are")
    print("    returned with them: strictly ordered %s, exact %s. A literal at the"
          % (ordered, exact))
    print("    root fails this on any machine; the span below cannot say that.")
    assert ordered and exact
    print("    full move graph %d undirected edges at mean degree %.1f, and that"
          % (pr["full_graph_edges"], pr["full_degree"]))
    print("    equals the DIRECTED arc count because the relation is bipartite by")
    print("    side to move: reciprocal arc pairs counted = %d, so symmetrising"
          % pr["n_reciprocal_arcs"])
    print("    doubles the nonzeros exactly. Counted, not argued.")
    assert pr["n_reciprocal_arcs"] == 0
    assert pr["full_graph_edges"] == sp["n_edges"]
    print("    THE PRICE IS A SPAN, NOT A POINT. A rate is a wall clock: two runs of")
    print("    this function on two machines gave %.3f and %.3f ms/edge, so the full"
          % (pr["observed_lo_ms"], pr["observed_hi_ms"]))
    print("    graph is %.1f h to %.1f h single-core (this run alone would say %.1f)."
          % (pr["span_lo_hours"], pr["span_hi_hours"], pr["lower_hours"]))
    print("    this run's rate is %s the recorded span, and %s the sanity band."
          % ("INSIDE" if pr["rate_in_span"] else "OUTSIDE",
             "inside" if pr["rate_in_band"] else "OUTSIDE"))
    print("    The span is accumulated evidence, not a check: it was breached three")
    print("    times in one afternoon on ONE machine as other work finished, which")
    print("    is the measurement of how little a single timing is worth. What")
    print("    discriminates a fabricated clock is the two lines above -- the raw")
    print("    reads, and the fact that the next run prints a different number.")
    assert pr["rate_in_band"]
    print("    A LOWER BOUND ON A LOWER BOUND: lower once because the rate comes")
    print("    from ball degree %.1f (about %.0f transport cells) while the full"
          % (pr["ball_degree"], pr["cells_ball"]))
    print("    graph runs at degree %.1f (about %.0f), and lower again because the"
          % (pr["full_degree"], pr["cells_full"]))
    print("    slowest machine that exists is not among the two that were timed.")
    assert pr["seconds"] > 0.0 and pr["n_attempted"] > 0
    assert abs(pr["ms_per_edge"]
               - 1000.0 * pr["seconds"] / pr["n_attempted"]) < 1e-9
    print("    RETIRED, kept on the page with its reason, and the reason is narrower")
    print("    than it was. An earlier revision printed 4.133 ms/edge, 22.9 s,")
    print("    >= 20,250 s = >= 5.6 h at ball degree 3.7 against full degree 26.6 as")
    print("    measured fact, from a scratch script that never entered the tree while")
    print("    this module imported only curvature_from_points and could not compute")
    print("    graph curvature at all. Struck for being UNREPRODUCIBLE, and correctly.")
    print("    NOT struck for being wrong: move_graph_price on a second box returned")
    print("    4.116 ms/edge, within 0.4% of the retired 4.133, so the figure")
    print("    reproduces. A revision of THIS section called it 1.9x wrong on one")
    print("    wall clock convicting another wall clock; that claim is withdrawn.")

    elapsed = time.time() - t0
    print("(k) BUDGET. elapsed %.1f s on CPU (bar: 300 s)." % elapsed)
    assert elapsed < 300.0, "self-check exceeded its 300 s budget: %.1f s" % elapsed
    print("ALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    demo()
