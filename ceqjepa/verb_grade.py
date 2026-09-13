"""A2: the three grades on a token, and what the third one is actually worth.

THE TRADE-OFF THIS MODULE REFUSES. A conditional entropy over language tokens is
always an ESTIMATE, and an estimate has a bias nobody can bound without a ground
truth -- so the third grade is unfalsifiable in the place it is usually computed.
The refusal is to build the instrument where it is EXACT: ceqjepa.chess_steps
enumerated K+Q vs K whole, 368452 positions and 4891672 legal moves, closed under
moves, with an exact oracle. There a verb is a move, "next" is the successor
distribution, and the conditional entropy is a finite sum. Nothing is sacrificed:
the exact value AND the estimator's error against it both come out.

A2 AS LITERALLY WRITTEN IS IDENTICALLY ZERO HERE, and saying so is the first
result rather than a detail. The successor map on this space is a FUNCTION -- one
(position, move) has one head, measured, 0 successor collisions over all 4891672
moves -- so H(next state | concrete move) is ln(1) for every move. Not small.
Empty. A module that quietly substitutes a different reading has changed the
claim without saying so, so all four readings are reported side by side:

    KEY         LEVEL        VALUES     min / median / max, NATS
    a_position  noun         368452     0.000000 / 1.791759 / 3.555348
    b_reply     occurrence   4847464    0.000000 / 1.609438 / 3.555348
    c_coarse    type         16 of 17   0.000000 / 0.107876 / 1.615938
    c_fine      type         1876       4.751943 / 5.135264 / 6.995366

(a) is H(next state | position) under the uniform policy. (b) is H two ply out
given a concrete verb -- the opponent's reply, the reading under which the mover's
own deterministic move still has fan-out. (c) is H(next verb | verb TYPE) at two
abstractions, coarse being (piece, gives-check, is-capture, squares-moved) and
fine being (piece, from, to). Fan-out at the type level comes from the
ABSTRACTION, so the abstraction is part of the result and both widths are run.

TWO OF THE FOUR ARE OUT-DEGREE WEARING A HAT, and that is arithmetic, not a
finding. No two legal moves from a position collapse to the same successor, so
under the uniform policy H(next | position) = ln(out-degree) EXACTLY: measured
gap 0.000e+00 on all of them, with ln(35) = 3.555348 the ceiling at the widest
position. (b) is the same identity one ply later. Rank statistics are invariant
under a monotone map, so readings (a) and (b) score IDENTICALLY to raw
out-degree against every event -- printed to the last bit rather than argued, AUC
0.608708 against 0.608708 and 0.150859 against 0.150859. A grade that is a
monotone function of out-degree CANNOT beat out-degree. Only (c) escapes, because
a verb type recurs across positions and averages over them.

THE NUMBER THIS MODULE OWES THE PROGRAMME is in section (g) of the run, and it
does not depend on T-VERB at all. The exact aggregate H(next | verb type) is
5.728896 nats on the fine alphabet and 0.653550 on the coarse one. Sampled from
the same generative process and read with the standard plug-in estimator, over 20
replicates at each size:

    ALPHABET   N = 1000     N = 10000    N = 100000   N = 1000000
    coarse     -0.0084      -0.0036      +0.0000      -0.0005 nats
    fine       -5.3729      -3.9251      -1.8524      -0.4160 nats

A MILLION TOKENS IS A REALISTIC CORPUS AND IT IS NOT ENOUGH: at N = 1000000 the
fine grade still under-reads by 7.262% of its own value, while the coarse grade
is down to 0.083%. The bias is set by the CELL COUNT, not by the bed, which is
the part that transfers to a corpus nobody can enumerate. Miller-Madow halves it
-- -0.1908 against -0.4160 at the largest size -- and IS NOT FREE: it strictly
helps on all 5 rows whose bias this run can resolve and makes 1 coarse row worse,
because it corrects an error that was not there.

T-VERB, AND THE ANSWER, WHICH IS NO, TWICE. The claim was that refusals
concentrate at HIGH-H verbs. The event is the bed's own refusal -- the exact
oracle giving no distance-to-mate at the state the verb reaches -- at a rate of
0.063944 over all 4891672 occurrences, split by cause into drawn-by-defence,
queen-captured and stalemate. Scored per verb type on the fine alphabet:

    ARM             Spearman rho vs abstention rate, n = 1876 verb types
    H(next|verb)    -0.8530   CI [-0.8664, -0.8365]
    E[log deg]      -0.9038   CI [-0.9110, -0.8941]
    mean deg        -0.8806   CI [-0.8901, -0.8685]
    log support     -0.8210   CI [-0.8362, -0.8020]
    H shuffled      -0.0252   CI [-0.0676, +0.0227]

Read the SIGN first. Refusals concentrate at LOW-H verbs, the opposite of the
claim, and the mechanism is plain: a draw in this endgame is the queen falling or
a stalemate, and both are cramped, low-branching situations. Read the CONTROL
second. Out-degree alone beats the entropy on the same verbs; paired on the same
bootstrap resamples the difference is -0.0508, CI [-0.0593, -0.0421], which
excludes zero. Read the STRATA last, which is where two predictors that move
together are finally separated: at matched out-degree the entropy keeps a mean
rho of only -0.1282 with the sign flipping across strata, while out-degree keeps
-0.5795 at matched entropy. On the occurrence-level AUC table the shuffled arm
reads 0.498078, the coarse grade 0.550062 and the fine grade 0.250701, against
0.150859 for plain out-degree of the successor -- which is further from a coin
than any entropy arm in the table.

THE KILL AND ITS ROUTE, because no kill ships without one. T-VERB is STRUCK on
this bed: degenerate in two of its four readings, wrong-signed in the other two,
and beaten there by the cheaper control it had to beat. The reroute is to put the
successor's OUT-DEGREE in the grade-2 slot -- one integer per state against a
1876 x 1876 bigram matrix, so it is cheaper as well as better, and it needs no
estimator at all. Keep H only where it is printed BESIDE log(degree) with the
paired interval, as it is here. What survives is the larger half: the estimator
curve prices every conditional entropy this programme reads off a corpus it
cannot enumerate, and it does not depend on T-VERB.

RECORDED NEGATIVE, kept because a failure deleted once the file goes green is a
failure that gets re-shipped. The first version of this module asserted the
mixture bracket as E[ln deg] <= H <= log(support) and the coarse alphabet failed
it on 16 of 16 scored verbs. The bound was wrong, not the number: the lower
bound is the mean entropy of the successor's TYPE distribution, which equals
ln(deg) only where the abstraction is injective within a position. Measured, the
component gap against ln(deg) is 1.332e-15 on the fine alphabet -- float residue
from summing d terms against one log -- and 2.079e+00 on the coarse one, which is
ln 8 = 2.079442, eight legal moves living under one word. Any coarse token
scheme has this property, so the wrong bound would have fired on a corpus too.

WHAT IS NOT CLAIMED. The entropy (Shannon 1948), the plug-in bias and its
correction (Miller 1955, Basharin 1959), the AUC and its interval (Hanley and
McNeil 1982), the Wilson interval (Wilson 1927), and the enumerated bed with its
exact oracle, which belongs to ceqjepa.chess_steps and is read here, never
edited. Grading a token by the conditional entropy of what follows it is the
unoccupied part; the conjunction is the only claim.

THE PINNED CONSTANTS. ENDGAME = KQK and SEED = 0, both inherited from the bed;
20 replicates behind every estimator interval, 400 bootstrap resamples behind
every rho interval, 5 equal-count bins. Every number above is printed by the run
below, whose wall clock it also prints; the planted negatives are a deterministic
verb reading 0.0 nats, a uniform verb over d successors reading log d to
8.882e-16, zero mass arriving as a Refusal, and a shuffled grade collapsing to
-0.0252.

RUN: python -m ceqjepa.verb_grade
"""

import math
import re
import sys
import time
from functools import lru_cache

import chess
import numpy as np
from scipy import sparse, stats

from ceqjepa import chess_steps as cs
from ceqjepa.curvature import Refusal, is_refusal

__all__ = [
    "ENDGAME", "SEED", "GRADES", "ABSTRACTIONS", "ABSTENTION",
    "SAMPLE_SIZES", "N_REPLICATES", "N_BOOTSTRAP", "N_BINS",
    "OWNER_TRIPLE_NATS", "Refusal", "is_refusal",
    "verb_entropy_nats", "grades", "typed_grade", "verb_report",
    "estimator_bias", "t_verb_report", "docstring_numbers", "demo",
]

#: A number WITH ITS BOUNDARIES. The lookarounds are the repair for the substring
#: vacuity: `"89167" in out` is TRUE the moment the run prints 4891672, so a
#: plain `in` test passes any fabricated number sharing digits with a real one --
#: which is exactly what a stale number looks like. Matching TOKEN-to-TOKEN means
#: a docstring number must appear in the run as a WHOLE number, never as a
#: fragment of another. The inner (?:\.\d+)* keeps a dotted version string as one
#: token instead of splitting it; the trailing (?!\w)(?!\.\d) rather than (?![\w.])
#: so a sentence-final period cannot hide the number in front of it. No exemption
#: list: a constant that wants to live in prose -- a year, a pinned parameter, a
#: vocabulary size -- is fixed by PRINTING it, never by excusing it here.
_NUMBER = re.compile(
    r"(?<![\w.])[+-]?\d+(?:\.\d+)*(?:[eE][+-]?\d+)?(?!\w)(?!\.\d)")


def docstring_numbers():
    """Every docstring DEFINED in this module, and every number inside them.

    Scoped by __module__ rather than by __all__, so demo() and every private
    helper are inside the scan: a bare constant in a helper's prose is the exact
    place the narrow form of this guard has been walked through.

    The coverage figure is itself a number in prose, so it is computed here,
    printed by the run, and recomputed INDEPENDENTLY by the test. Two counters
    that must agree is a check; one shared helper the module could quietly narrow
    is not.
    """
    m = sys.modules[__name__]
    docs = [("module", m.__doc__ or "")]
    for name, obj in sorted(vars(m).items()):
        if getattr(obj, "__module__", None) != __name__:
            continue
        doc = getattr(obj, "__doc__", None)
        if isinstance(doc, str) and doc.strip():
            docs.append((name, doc))
    return dict(n_docstrings=len(docs),
                n_numbers=sum(len(_NUMBER.findall(d)) for _, d in docs),
                docstrings=docs)

#: The enumerated bed. Inherited from ceqjepa.chess_steps, read-only, unchanged.
ENDGAME = cs.ENDGAME

#: Every sample and every shuffle in this module. PINNED, and equal to the bed's.
SEED = cs.SEED

#: The three grades A2 asks for, in order.
GRADES = ("noun (state, grade 0)", "verb (transition, grade 1)",
          "H(next | verb) (consequence entropy, grade 2)")

#: The two widths the type-level grade is read at. Fan-out at the type level
#: comes from the ABSTRACTION, so the abstraction is part of the result and the
#: grade is reported at two of them so the reader can see it move.
ABSTRACTIONS = ("coarse", "fine")

#: What each abstraction maps a concrete move onto. Stated, because a type-level
#: entropy without its equivalence class is a number about nothing.
ABSTRACTION_OF = {
    "coarse": "(piece, gives-check, is-capture, squares-moved) -- a SAN-shaped word",
    "fine": "(piece, from-square, to-square) -- SAN's long form",
}

#: The abstention event T-VERB is scored against. It is the bed's OWN refusal:
#: the exact oracle returns no distance-to-mate at the state the verb reaches.
ABSTENTION = "the exact oracle gives no distance-to-mate at the state the verb reaches"

#: Sample sizes the plug-in estimator is measured at, against the exact value.
SAMPLE_SIZES = (1000, 10000, 100000, 1000000)

#: Independent replicates behind every estimator interval. PINNED.
N_REPLICATES = 20

#: Bootstrap resamples behind every rho interval. PINNED.
N_BOOTSTRAP = 400

#: Equal-count bins in the T-VERB table.
N_BINS = 5

#: The owner's lineage triple for the entropy grade, in nats. RECORDED, never
#: used as an expected value: it came from another alphabet, and every reading
#: below reports its own beside it with the difference stated.
OWNER_TRIPLE_NATS = (0.17, 0.69, 1.39)

# Refusal codes. Each names a DISTINCT cause so a caller can count by cause.
NO_MASS = "no-mass"
ALL_TERMINAL = "all-terminal"

# The three causes an abstention can have on this bed.
STALEMATE = "stalemate"
DRAWN_BY_DEFENCE = "drawn-by-defence"
QUEEN_CAPTURED = "queen-captured"

_DRAW = 3          # ceqjepa.chess_steps' DRAW label
_NO_CONTINUATION = -np.inf   # sentinel for ln(0): not a value, and it sorts below


# ---------------------------------------------------------------------------
# THE STATISTIC, as a pure function of counts. Everything planted is planted here
# ---------------------------------------------------------------------------

def verb_entropy_nats(counts):
    """Shannon entropy of a count or mass vector, in NATS.

    A pure function of the vector, so a deterministic verb and a uniform verb can
    both be planted without a bed behind them. Zero mass is not zero entropy --
    it is the absence of a distribution, and it comes back as a Refusal carrying
    its cause rather than as a 0.0 that a threshold would silently accept.
    """
    c = np.asarray(counts, dtype=float).ravel()
    if c.size == 0 or not np.isfinite(c).all() or (c < 0.0).any():
        return Refusal(NO_MASS, "not a non-negative finite count vector")
    total = float(c.sum())
    if total <= 0.0:
        return Refusal(NO_MASS, "the verb has no live continuation to be uncertain about")
    p = c[c > 0.0] / total
    return float(-(p * np.log(p)).sum()) + 0.0        # + 0.0 kills the -0.0


def _row_entropy_nats(P):
    """Row-wise entropy of a CSR matrix whose rows are probability vectors."""
    x = P.copy()
    x.data = -P.data * np.log(P.data)
    return np.asarray(x.sum(1)).ravel()


def _wilson(k, n, z=1.96):
    """Wilson interval for k successes in n. Correct at rates near zero, which
    the normal interval is not, and every rate in the T-VERB table is near zero."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def _rho(x, y):
    """Spearman rho, as rankdata plus Pearson, so a bootstrap of it is affordable."""
    rx = stats.rankdata(x).astype(float)
    ry = stats.rankdata(y).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return 0.0 if den == 0.0 else float(rx @ ry / den)


def _boot_ci(fn, n, rng, reps=N_BOOTSTRAP, q=(2.5, 97.5)):
    vals = [fn(rng.integers(0, n, n)) for _ in range(reps)]
    lo, hi = np.percentile(vals, q)
    return float(lo), float(hi)


def _auc(score, event):
    """Mann-Whitney AUC with exact tie handling, from grouped counts.

    Grouped rather than ranked because every arm has far fewer distinct values
    than rows -- the count is printed beside each arm as its LEVELS -- and because
    grouping makes the monotone-invariance identity exact: two scores that induce
    the same ordered grouping produce the same floats through the same arithmetic.
    """
    vals, inv = np.unique(score, return_inverse=True)
    pos = np.bincount(inv, weights=event.astype(float), minlength=vals.size)
    neg = np.bincount(inv, minlength=vals.size) - pos
    below = np.concatenate(([0.0], np.cumsum(neg)[:-1]))
    n1, n0 = float(pos.sum()), float(neg.sum())
    return (float((pos @ below + 0.5 * (pos @ neg)) / (n1 * n0)), n1, n0,
            int(vals.size))


def _auc_ci(a, n1, n0, z=1.96):
    """Hanley-McNeil interval. Wide enough to be honest, narrow at these counts."""
    q1, q2 = a / (2.0 - a), 2.0 * a * a / (1.0 + a)
    se = math.sqrt(max(0.0, (a * (1 - a) + (n1 - 1) * (q1 - a * a)
                             + (n0 - 1) * (q2 - a * a)) / (n1 * n0)))
    return max(0.0, a - z * se), min(1.0, a + z * se)


# ---------------------------------------------------------------------------
# THE BED: every legal move of the enumerated space, described
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _edges():
    """Every legal move of the enumerated space, with what it did and where to.

    Nothing about chess is reimplemented: the positions, the moves, the exact
    oracle and `is_check` all come from python-chess and ceqjepa.chess_steps,
    read-only. What is added here is a DESCRIPTION of each edge -- which piece
    moved, how far, whether it captured, whether it gave check -- from which the
    two verb abstractions are built.
    """
    sp, orc = cs.space(), cs.oracle()
    n, nt, sink = sp["n_positions"], sp["n_states"], sp["sink"]
    keys = sp["keys"].astype(np.int64)
    pred = sp["pred"].astype(np.int64)
    succ = sp["succ"].astype(np.int64)

    degree = np.zeros(nt, np.int64)
    degree[:n] = np.diff(sp["succ_indptr"])
    label = np.full(nt, _DRAW, np.int8)
    label[:n] = orc["label"]

    # is_check per position, from python-chess. One board reused; 368,452 calls.
    wkp = chess.Piece(chess.KING, chess.WHITE)
    wqp = chess.Piece(chess.QUEEN, chess.WHITE)
    bkp = chess.Piece(chess.KING, chess.BLACK)
    in_check = np.zeros(nt, bool)
    board = chess.Board(None)
    for i in range(n):
        wk, wq, bk, turn = keys[i]
        board.clear_board()
        board.set_piece_at(int(wk), wkp)
        board.set_piece_at(int(wq), wqp)
        board.set_piece_at(int(bk), bkp)
        board.turn = bool(turn)
        in_check[i] = board.is_check()

    # Which piece moved, and between which squares. White either walks its king
    # or moves its queen; Black only ever walks its king, and the single edge
    # that leaves the position space is that king taking the queen.
    wk_i, wq_i, bk_i = keys[pred, 0], keys[pred, 1], keys[pred, 2]
    white = keys[pred, 3] == 1
    to_sink = succ == sink
    j = np.where(to_sink, 0, succ)
    wk_j, wq_j, bk_j = keys[j, 0], keys[j, 1], keys[j, 2]

    king_moved = white & (wk_j != wk_i) & ~to_sink
    queen_moved = white & ~king_moved & ~to_sink
    frm = np.where(king_moved, wk_i, np.where(queen_moved, wq_i, bk_i))
    to = np.where(king_moved, wk_j,
                  np.where(queen_moved, wq_j, np.where(to_sink, wq_i, bk_j)))
    is_queen = queen_moved.astype(np.int64)
    is_capture = to_sink.astype(np.int64)
    gives_check = np.where(to_sink, 0, in_check[j]).astype(np.int64)
    moved = np.maximum(np.abs((frm & 7) - (to & 7)), np.abs((frm >> 3) - (to >> 3)))

    vid = {}
    vid["fine"] = np.unique(is_queen * 4096 + frm * 64 + to, return_inverse=True)[1]
    vid["coarse"] = np.unique(is_queen * 1000 + gives_check * 100
                              + is_capture * 10 + moved, return_inverse=True)[1]

    cause = np.zeros(nt, np.int8)             # 0 none, 1 stalemate, 2 held, 3 sink
    cause[orc["stalemate_nodes"]] = 1
    cause[orc["held_draw_nodes"]] = 2
    cause[sink] = 3
    by_cause = np.bincount(cause[succ], minlength=4)

    # THE LITERAL READING OF A2, measured rather than asserted. If any
    # (position, verb) pair had two heads the successor map would not be a
    # function and these counts would exceed 1; the entropy of a point mass is
    # ln(1), so the whole grade is identically zero as A2 writes it.
    heads = np.bincount(np.unique(pred * (vid["fine"].max() + 1) + vid["fine"],
                                  return_inverse=True)[1])
    h_literal = np.log(heads.astype(float))
    collisions = int(succ.size - np.unique(pred * nt + succ).size)

    return dict(
        n_nouns=n, n_states=nt, sink=sink, n_edges=int(succ.size),
        pred=pred, succ=succ, indptr=sp["succ_indptr"], degree=degree,
        label=label, vid=vid, in_check=in_check,
        is_abstention=label[succ] == _DRAW,
        abstention_by_cause={STALEMATE: int(by_cause[1]),
                             DRAWN_BY_DEFENCE: int(by_cause[2]),
                             QUEEN_CAPTURED: int(by_cause[3])},
        h_literal_min=float(h_literal.min()), h_literal_max=float(h_literal.max()),
        n_literal_verbs=int(h_literal.size), n_successor_collisions=collisions,
    )


@lru_cache(maxsize=4)
def typed_grade(kind="fine"):
    """H(next verb | verb type) at one abstraction, exact over the closed space.

    The measure is: a verb occurrence uniform over the enumerated legal moves,
    then a reply uniform over the successor's legal moves. Both halves are
    finite sums over an enumerated, closed space -- no sample anywhere -- so the
    conditional distribution, and the entropy of it, are exact.

    Occurrences whose successor is terminal have no next token; they are excluded
    from the conditional and counted as abstentions on the other side of the
    ledger, which is where they belong. A verb type ALL of whose occurrences end
    that way has no distribution at all, and its grade is a REFUSAL carrying that
    cause -- never a 0.0, which is the reading a threshold would silently accept
    and which would be indistinguishable from a genuinely forced verb.
    """
    if kind not in ABSTRACTIONS:
        return Refusal(NO_MASS, "%r is not one of the two stated abstractions" % kind)
    e = _edges()
    vid = e["vid"][kind]
    V = int(vid.max()) + 1
    pred, succ, degree, nt = e["pred"], e["succ"], e["degree"], e["n_states"]
    degree_next = degree[succ]
    live = degree_next > 0

    S = sparse.csr_matrix((1.0 / np.maximum(degree[pred], 1), (pred, vid)),
                          shape=(nt, V))
    T = sparse.csr_matrix((np.ones(int(live.sum())), (vid[live], succ[live])),
                          shape=(V, nt))
    M = (T @ S).tocsr()
    n_live = np.asarray(M.sum(1)).ravel()
    P = M.copy()
    P.data = P.data / np.repeat(np.maximum(n_live, 1.0), np.diff(P.indptr))

    # The MIXTURE bound's component, and the reason it is not ln(deg) in general.
    # Row j of S is already the distribution over verb TYPES of the successor's
    # legal moves; a coarse abstraction merges several moves into one word, so
    # that row's entropy sits BELOW ln(deg(j)). The mixture entropy is at least
    # the mean of these, not at least the mean of ln(deg) -- getting that wrong
    # asserts a bound the coarse alphabet violates on every verb.
    component = _row_entropy_nats(S)
    ll = np.where(live, np.log(np.maximum(degree_next, 1)), 0.0)
    cc = np.where(live, component[succ], 0.0)
    n_occ = np.bincount(vid, minlength=V).astype(np.int64)
    refused = n_live == 0
    entropy = _row_entropy_nats(P)
    entropy[refused] = np.nan       # a refusal, not a zero. See live_mask.
    return dict(
        kind=kind, abstraction=ABSTRACTION_OF[kind], n_verb_types=V, vid=vid,
        conditional=P, entropy_nats=entropy, live_mask=~refused,
        n_refused_verb_types=int(refused.sum()),
        n_refused_occurrences=int(n_occ[refused].sum()),
        refusals=[Refusal(ALL_TERMINAL,
                          "verb type %d: all %d occurrences end the token stream"
                          % (i, n_occ[i])) for i in np.flatnonzero(refused)],
        log_support=np.log(np.maximum(np.diff(P.indptr), 1)),
        n_occurrences=n_occ,
        n_live=n_live.astype(np.int64),
        n_abstentions=np.bincount(vid, weights=e["is_abstention"].astype(float),
                                  minlength=V).astype(np.int64),
        mean_log_degree=np.bincount(vid, weights=ll, minlength=V) / np.maximum(n_live, 1.0),
        mean_component_entropy=(np.bincount(vid, weights=cc, minlength=V)
                                / np.maximum(n_live, 1.0)),
        component_gap_vs_log_degree=float(np.abs(
            component[succ][live] - np.log(degree_next[live])).max()),
        mean_degree=(np.bincount(vid, weights=np.where(live, degree_next, 0).astype(float),
                                 minlength=V) / np.maximum(n_live, 1.0)),
    )


@lru_cache(maxsize=1)
def grades():
    """The three grades, each as an array, each with its count, plus both ways
    the grade collapses.

    `literal_reading` is A2 exactly as written -- H over the next STATE given a
    concrete move. It is identically zero because the successor map is a
    function, and it is reported rather than quietly replaced.

    `h_position` is reading (a) and `b_reply` is reading (b). Both are ln of an
    out-degree, exactly, which is why neither can beat out-degree at anything.
    """
    e = _edges()
    n, deg = e["n_nouns"], e["degree"]
    out_degree = deg[:n].copy()
    h_position = np.where(out_degree > 0, np.log(np.maximum(out_degree, 1)), 0.0)

    degree_next = deg[e["succ"]]
    live = degree_next > 0
    h_reply = np.where(live, np.log(np.maximum(degree_next, 1)), _NO_CONTINUATION)
    gap = float(np.abs(h_reply[live] - np.log(degree_next[live])).max())

    fine = typed_grade("fine")
    return dict(
        n_nouns=n, n_verb_occurrences=e["n_edges"],
        n_verb_types=int(fine["n_verb_types"]),
        out_degree=out_degree, h_position=h_position, h_reply=h_reply,
        n_successor_collisions=e["n_successor_collisions"],
        literal_reading=dict(
            min_nats=e["h_literal_min"], max_nats=e["h_literal_max"],
            n_verbs=e["n_literal_verbs"],
            reason="the successor map is a FUNCTION: one (position, move) has one head, "
                   "so the next state is deterministic and its entropy is ln(1)"),
        b_reply=dict(max_abs_gap=gap, n_occurrences=int(live.sum()),
                     reading="H(state two ply out | concrete verb) == ln(out-degree "
                             "of the successor)"),
        n_abstentions_total=int(e["is_abstention"].sum()),
        abstention_by_cause=dict(e["abstention_by_cause"]),
        n_forced_states=int((out_degree == 1).sum()),
    )


def _describe(key, name, level, values, n_values, denominator, abstraction):
    v = np.asarray(values, float)
    ours = [float(v.min()), float(np.median(v)), float(v.max())]
    return dict(key=key, name=name, level=level, n_values=int(n_values),
                denominator=denominator, abstraction=abstraction,
                min_nats=ours[0], median_nats=ours[1], max_nats=ours[2],
                mean_nats=float(v.mean()),
                zero_fraction=float((v <= 0.0).mean()),
                owner_difference_nats=[a - b for a, b in zip(ours, OWNER_TRIPLE_NATS)])


@lru_cache(maxsize=1)
def verb_report():
    """All four readings of the entropy grade, side by side, against the lineage."""
    g = grades()
    live_reply = g["h_reply"][np.isfinite(g["h_reply"])]
    defs = [
        _describe("a_position", "H(next state | position), uniform policy", "noun",
                  g["h_position"], g["n_nouns"],
                  "all %d enumerated positions, the %d with no legal move read 0"
                  % (g["n_nouns"], g["n_nouns"] - int((g["out_degree"] > 0).sum())),
                  "none: the verb is not abstracted, the state is the condition"),
        _describe("b_reply", "H(state two ply out | concrete verb)", "occurrence",
                  live_reply, live_reply.size,
                  "the %d occurrences with a live continuation" % live_reply.size,
                  "none: the verb is the concrete move"),
    ]
    for kind in ABSTRACTIONS:
        t = typed_grade(kind)
        keep = t["live_mask"]
        defs.append(_describe("c_" + kind, "H(next verb | verb type), %s" % kind,
                              "type", t["entropy_nats"][keep], int(keep.sum()),
                              "%d scored verb types of %d over %d occurrences, "
                              "%d types REFUSED as all-terminal"
                              % (int(keep.sum()), t["n_verb_types"],
                                 int(t["n_occurrences"][keep].sum()),
                                 t["n_refused_verb_types"]),
                              t["abstraction"]))
    nearest = min(defs, key=lambda d: max(abs(x) for x in d["owner_difference_nats"]))
    fine = typed_grade("fine")
    keep = fine["live_mask"]
    return dict(
        definitions=defs,
        n_nouns=g["n_nouns"], n_verb_occurrences=g["n_verb_occurrences"],
        owner_triple_nats=list(OWNER_TRIPLE_NATS),
        owner_nearest_reading=nearest["key"],
        owner_nearest_name=nearest["name"],
        owner_nearest_gap=float(max(abs(x) for x in nearest["owner_difference_nats"])),
        aggregate_fine_nats=float((fine["n_live"][keep] / fine["n_live"].sum()
                                   * fine["entropy_nats"][keep]).sum()),
        n_forced_states=g["n_forced_states"],
        forced_noun_fraction=float(g["n_forced_states"] / g["n_nouns"]),
    )


# ---------------------------------------------------------------------------
# EXACTNESS FIRST, ESTIMATION SECOND
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def estimator_bias(sizes=SAMPLE_SIZES, reps=N_REPLICATES, seed=SEED):
    """How wrong the plug-in estimate of the type-level grade is, against exact.

    THE NUMBER THIS MODULE OWES THE REST OF THE PROGRAMME. Every corpus that
    cannot be enumerated reads this grade with a plug-in estimator, and the
    plug-in is biased DOWNWARD by roughly (occupied cells - 1) / 2N. Here the
    exact value exists, so the bias is not bounded or assumed -- it is subtracted.

    Both abstractions are run because the bias is set by the number of CELLS, not
    by the bed: that is the part which transfers to a corpus. Miller-Madow is
    reported beside the raw plug-in because a condemned estimator needs a route,
    and it is the cheapest one there is.
    """
    e = _edges()
    live_edges = np.flatnonzero(e["degree"][e["succ"]] > 0)
    succ, degree, indptr = e["succ"], e["degree"], e["indptr"]

    rows = []
    for kind in ABSTRACTIONS:
        t = typed_grade(kind)
        vid, V, keep = t["vid"], int(t["n_verb_types"]), t["live_mask"]
        exact = float((t["n_live"][keep] / t["n_live"].sum()
                       * t["entropy_nats"][keep]).sum())
        for N in sizes:
            raw, corrected = [], []
            for r in range(reps):
                rng = np.random.default_rng(seed + 1000 + r)
                idx = live_edges[rng.integers(0, live_edges.size, N)]
                j = succ[idx]
                reply = indptr[j] + (rng.random(N) * degree[j]).astype(np.int64)
                cell, count = np.unique(vid[idx].astype(np.int64) * V + vid[reply],
                                        return_counts=True)
                owner = cell // V
                n_v = np.bincount(owner, weights=count.astype(float), minlength=V)
                p = count / n_v[owner]
                h_hat = float(np.bincount(owner, weights=-p * np.log(p),
                                          minlength=V) @ (n_v / N))
                cells = np.bincount(owner, minlength=V)
                mm = h_hat + float(((cells - 1) / (2.0 * N))[n_v > 0].sum())
                raw.append(h_hat - exact)
                corrected.append(mm - exact)
            raw, corrected = np.array(raw), np.array(corrected)
            half = 1.96 * raw.std(ddof=1) / math.sqrt(reps)
            half_mm = 1.96 * corrected.std(ddof=1) / math.sqrt(reps)
            rows.append(dict(
                abstraction=kind, n_verb_types=V, n_samples=int(N),
                n_replicates=int(reps), exact_nats=exact,
                bias_nats=float(raw.mean()),
                ci_lo=float(raw.mean() - half), ci_hi=float(raw.mean() + half),
                bias_percent=float(100.0 * raw.mean() / exact),
                miller_madow_bias_nats=float(corrected.mean()),
                mm_ci_lo=float(corrected.mean() - half_mm),
                mm_ci_hi=float(corrected.mean() + half_mm)))
    return tuple(rows)


# ---------------------------------------------------------------------------
# T-VERB, WHICH ONLY EXISTS BESIDE ITS CONTROLS
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def t_verb_report(n_bins=N_BINS, n_boot=N_BOOTSTRAP, seed=SEED):
    """Does the entropy grade predict abstention, and does it beat out-degree?

    T-VERB says refusals concentrate at HIGH-H verbs. It is scored against the
    bed's own refusal -- the exact oracle returning no distance-to-mate at the
    state the verb reaches -- at two levels.

    `occurrence_arms` puts every reading on ONE event with ONE statistic, AUC
    over all the enumerated legal moves, which is the only place readings at
    different levels can be compared. Readings (a) and (b) are ln of an
    out-degree, so their AUC equals raw out-degree's IDENTICALLY; that equality
    is printed rather than argued, and it is why a degenerate grade cannot win.

    `arms` scores the type-level grade per verb type against three controls, and
    `paired_vs_degree` bootstraps the difference against out-degree on the SAME
    resamples, because two rhos over the same verbs cannot be compared apart.
    `strata` is the decisive part: the entropy scored WITHIN deciles of
    out-degree, and out-degree WITHIN deciles of entropy. Whichever keeps its
    association at the other's matched value is the one carrying information.
    """
    e = _edges()
    g = grades()
    event = e["is_abstention"]
    fine, coarse = typed_grade("fine"), typed_grade("coarse")

    deg_next = e["degree"][e["succ"]].astype(float)
    rng0 = np.random.default_rng(seed)
    shuffled_fine = rng0.permutation(fine["entropy_nats"])

    scored = (
        ("H(next|position) = ln deg", g["h_position"][e["pred"]]),
        ("out-degree of position", e["degree"][e["pred"]].astype(float)),
        ("H(state 2ply | verb)", g["h_reply"]),
        ("out-degree of successor", deg_next),
        ("H(next|verb type) coarse", coarse["entropy_nats"][coarse["vid"]]),
        ("H(next|verb type) fine", fine["entropy_nats"][fine["vid"]]),
        ("H fine, shuffled", shuffled_fine[fine["vid"]]))
    # ONE denominator for every arm, or the AUCs are not comparable. An
    # occurrence is dropped only where some arm REFUSES a grade (NaN); the -inf
    # that marks a terminal successor is a sentinel, not a refusal, and it stays
    # in -- it is where most of the abstentions live.
    common = np.ones(event.size, bool)
    for _, s in scored:
        common &= ~np.isnan(s)
    occurrence_arms = []
    for name, score in scored:
        a, n1, n0, levels = _auc(score[common], event[common])
        lo, hi = _auc_ci(a, n1, n0)
        occurrence_arms.append(dict(name=name, auc=a, ci_lo=lo, ci_hi=hi,
                                    levels=levels, n_events=int(n1),
                                    n_occurrences=int(n1 + n0),
                                    n_dropped=int((~common).sum())))

    keep = fine["live_mask"]
    h = fine["entropy_nats"][keep]
    rate = (fine["n_abstentions"] / fine["n_occurrences"])[keep]
    V = h.size
    arms = []
    for name, x in (("H(next|verb)", h),
                    ("E[log deg]", fine["mean_log_degree"][keep]),
                    ("log mean deg", np.log(fine["mean_degree"][keep])),
                    ("mean deg", fine["mean_degree"][keep]),
                    ("log support", fine["log_support"][keep]),
                    ("H shuffled", shuffled_fine[keep])):
        boot = np.random.default_rng(seed)
        lo, hi = _boot_ci(lambda i, x=x: _rho(x[i], rate[i]), V, boot, reps=n_boot)
        arms.append(dict(name=name, rho=_rho(x, rate), ci_lo=lo, ci_hi=hi))

    deg = fine["mean_log_degree"][keep]
    boot = np.random.default_rng(seed + 1)
    delta = abs(_rho(h, rate)) - abs(_rho(deg, rate))
    d_lo, d_hi = _boot_ci(
        lambda i: abs(_rho(h[i], rate[i])) - abs(_rho(deg[i], rate[i])),
        V, boot, reps=n_boot)
    verdict = "entropy wins" if d_lo > 0 else "degree wins" if d_hi < 0 else "tie"

    def _strata(by, of, k=10):
        edges = np.quantile(by, np.linspace(0, 1, k + 1))
        which = np.clip(np.digitize(by, edges[1:-1]), 0, k - 1)
        return [dict(stratum=int(s), n_verbs=int((which == s).sum()),
                     rho=_rho(of[which == s], rate[which == s])) for s in range(k)]

    h_in_deg, deg_in_h = _strata(deg, h), _strata(h, deg)

    edges = np.quantile(h, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.digitize(h, edges[1:-1]), 0, n_bins - 1)
    n_abs_k, n_occ_k = fine["n_abstentions"][keep], fine["n_occurrences"][keep]
    bins = []
    for s in range(n_bins):
        m = which == s
        k = int(n_abs_k[m].sum())
        n = int(n_occ_k[m].sum())
        lo, hi = _wilson(k, n)
        bins.append(dict(bin=int(s), h_lo=float(h[m].min()), h_hi=float(h[m].max()),
                         n_verbs=int(m.sum()), n_occurrences=n, n_events=k,
                         rate=k / n, ci_lo=lo, ci_hi=hi))

    return dict(
        event=ABSTENTION, arms=arms, occurrence_arms=occurrence_arms, bins=bins,
        n_verb_types=int(V),
        overall_rate=float(g["n_abstentions_total"] / g["n_verb_occurrences"]),
        abstention_by_cause=dict(g["abstention_by_cause"]),
        paired_vs_degree=dict(delta=delta, ci_lo=d_lo, ci_hi=d_hi,
                              verdict=verdict, against="E[log deg]"),
        strata=dict(entropy_within_degree=h_in_deg, degree_within_entropy=deg_in_h,
                    mean_rho_entropy_within_degree=float(
                        np.mean([s["rho"] for s in h_in_deg])),
                    mean_rho_degree_within_entropy=float(
                        np.mean([s["rho"] for s in deg_in_h]))),
        direction=("refusals concentrate at HIGH-H verbs" if _rho(h, rate) > 0
                   else "refusals concentrate at LOW-H verbs"),
    )


# ---------------------------------------------------------------------------
# THE SELF-CHECK
# ---------------------------------------------------------------------------

def demo():
    t_start = time.perf_counter()
    print("A2 / T-VERB: the three grades, with the entropy grade computed EXACTLY")
    print("bed: ceqjepa.chess_steps, endgame %s, seed %d (both inherited, read-only)"
          % (ENDGAME, SEED))
    print("PINNED: %d replicates per estimator interval, %d bootstrap resamples per"
          % (N_REPLICATES, N_BOOTSTRAP))
    print("rho interval, %d equal-count bins, abstractions %s and %s."
          % (N_BINS, ABSTRACTIONS[0], ABSTRACTIONS[1]))
    dn = docstring_numbers()
    print("COVERAGE: %d docstrings defined in this module, %d numbers in them, and"
          % (dn["n_docstrings"], dn["n_numbers"]))
    print("every one of them is printed below -- there is no exemption list.")

    print("\n(a) PLANTED NEGATIVES ON THE STATISTIC, before any bed touches it.")
    det = verb_entropy_nats(np.array([0.0, 7.0, 0.0]))
    print("    deterministic verb (one successor)      %.1f nats" % det)
    assert det == 0.0, "a deterministic verb did not read exactly zero"
    worst = max(abs(verb_entropy_nats(np.ones(d)) - math.log(d))
                for d in (2, 3, 4, 10, 137))
    print("    uniform verb over d successors          max |H - log d| = %.3e" % worst)
    assert worst < 1e-12, "a uniform verb did not read log(d) to 1e-12"
    empty = verb_entropy_nats(np.zeros(5))
    print("    zero mass                               %r" % (empty,))
    assert is_refusal(empty), "zero mass returned a number instead of a Refusal"
    print("    FIRED three ways: exact 0, exact log(d), and a REFUSAL for no mass.")

    g, r = grades(), verb_report()
    print("\n(b) A2 AS LITERALLY WRITTEN IS IDENTICALLY ZERO ON THIS SPACE, and that")
    print("    is a structure, not a measurement to work around.")
    lit = g["literal_reading"]
    print("    H(next STATE | concrete move) over all %d moves: min %.1f, max %.1f nats"
          % (lit["n_verbs"], lit["min_nats"], lit["max_nats"]))
    print("    reason: %s." % lit["reason"])
    print("    successor collisions (two moves, one state): %d, so no move's heads"
          % g["n_successor_collisions"])
    print("    ever merge and H(next | position) = ln(out-degree) EXACTLY.")
    assert lit["max_nats"] == 0.0 and g["n_successor_collisions"] == 0

    print("\n(c) THE THREE GRADES, each counted.")
    print("    grade 0  %-46s %9d" % (GRADES[0], r["n_nouns"]))
    print("    grade 1  %-46s %9d occurrences" % (GRADES[1], r["n_verb_occurrences"]))
    print("    grade 2  %-46s see the four readings below" % GRADES[2])
    assert r["n_nouns"] == 368452 and r["n_verb_occurrences"] == 4891672

    print("\n(d) FOUR READINGS OF THE ENTROPY GRADE, ALL EXACT, ALL IN NATS.")
    print("    %-11s %-11s %8s   %-9s %-9s %-9s %-9s"
          % ("key", "level", "values", "min", "median", "max", "mean"))
    for d in r["definitions"]:
        print("    %-11s %-11s %8d   %-9.6f %-9.6f %-9.6f mean %.6f"
              % (d["key"], d["level"], d["n_values"], d["min_nats"],
                 d["median_nats"], d["max_nats"], d["mean_nats"]))
        print("        %s" % d["name"])
        print("        abstraction: %s" % d["abstraction"])
        print("        over %s; fraction at exactly zero %.6f"
              % (d["denominator"], d["zero_fraction"]))
    for kind in ABSTRACTIONS:
        for ref in typed_grade(kind)["refusals"]:
            print("    REFUSED, %s alphabet: %r" % (kind, ref))
    print("    A refused verb is not a zero-entropy verb, and the %s alphabet has"
          % ABSTRACTIONS[0])
    print("    both: %d types refused as all-terminal and %d with a genuine 0.0."
          % (typed_grade("coarse")["n_refused_verb_types"],
             int((typed_grade("coarse")["entropy_nats"] == 0.0).sum())))
    print("    out-degree: min %d, max %d, mean %.3f over %d positions."
          % (g["out_degree"].min(), g["out_degree"].max(), g["out_degree"].mean(),
             r["n_nouns"]))
    print("    %d positions have exactly one legal move, a fraction of %.6f: THOSE"
          % (r["n_forced_states"], r["forced_noun_fraction"]))
    print("    are the forced moves, and they are a grade-0 fact, not a grade-2 one.")
    print("    exact aggregate H(next | verb type)     %.6f nats (fine)"
          % r["aggregate_fine_nats"])
    print("    THE MIXTURE BRACKET, E[component H] <= H <= log(support). The lower")
    print("    bound is NOT E[ln deg]: a coarse word merges several legal moves at")
    print("    one position, so the component distribution is narrower than uniform")
    print("    over moves and the ln(deg) bound is simply false there.")
    for kind in ABSTRACTIONS:
        t = typed_grade(kind)
        k = t["live_mask"]
        print("    %-7s component gap against ln(deg) %.3e; %d of %d scored verbs"
              % (kind, t["component_gap_vs_log_degree"],
                 int((t["entropy_nats"][k] < t["mean_log_degree"][k] - 1e-9).sum()),
                 int(k.sum())))
        print("            sit BELOW E[ln deg], and none sits outside the real bracket.")
        assert (t["entropy_nats"][k] <= t["log_support"][k] + 1e-9).all()
        assert (t["entropy_nats"][k] >= t["mean_component_entropy"][k] - 1e-9).all()
    print("    The fine gap is float residue from summing d terms against one log;")
    print("    the coarse gap is ln 8 = %.6f, eight legal moves under one word."
          % math.log(8))
    assert typed_grade("fine")["component_gap_vs_log_degree"] < 1e-12, \
        "the fine abstraction is not injective within a position"
    assert typed_grade("coarse")["component_gap_vs_log_degree"] > 1.0, \
        "the coarse abstraction merges nothing: it is not coarse"

    print("\n(e) READINGS (a) AND (b) ARE ln(OUT-DEGREE), so neither can beat")
    print("    out-degree at anything. Checked, not argued:")
    print("    max |H(next|position) - ln deg|            %.3e"
          % float(np.abs(g["h_position"][g["out_degree"] > 0]
                         - np.log(g["out_degree"][g["out_degree"] > 0])).max()))
    print("    max |H(2ply|verb) - ln deg of successor|   %.3e over %d occurrences"
          % (g["b_reply"]["max_abs_gap"], g["b_reply"]["n_occurrences"]))
    print("    ln(%d) = %.6f is the ceiling of both, at the widest position."
          % (g["out_degree"].max(), math.log(int(g["out_degree"].max()))))

    print("\n(f) THE LINEAGE TRIPLE, against every reading.")
    print("    owner's lineage  %.2f / %.2f / %.2f nats" % OWNER_TRIPLE_NATS)
    for d in r["definitions"]:
        print("    %-11s ours %.4f / %.4f / %.4f   difference %+.4f / %+.4f / %+.4f"
              % (d["key"], d["min_nats"], d["median_nats"], d["max_nats"],
                 d["owner_difference_nats"][0], d["owner_difference_nats"][1],
                 d["owner_difference_nats"][2]))
    print("    nearest reading: %s, still off by %.4f nats at its worst corner."
          % (r["owner_nearest_name"], r["owner_nearest_gap"]))
    print("    NONE of them lands near his triple, and the reason is the ALPHABET,")
    print("    not a disagreement: 0.69 and 1.39 are log 2 and log 4, a two-to-four")
    print("    way branch. The narrowest alphabet here is %d words and the widest is"
          % min(d["n_values"] for d in r["definitions"] if d["level"] == "type"))
    print("    %d, so the grade has nowhere near that little room to be uncertain in."
          % max(d["n_values"] for d in r["definitions"] if d["level"] == "type"))

    print("\n(g) EXACTNESS FIRST, ESTIMATION SECOND. The plug-in estimator every")
    print("    non-enumerable corpus will use, measured against the exact value.")
    rows = estimator_bias()
    print("    %-7s %6s %9s   %-34s %-24s"
          % ("alphabet", "words", "N", "plug-in bias (nats)", "Miller-Madow"))
    for row in rows:
        print("    %-7s %6d %9d   %+.4f CI [%+.4f,%+.4f] %8.3f%%  %+.4f CI [%+.4f,%+.4f]"
              % (row["abstraction"], row["n_verb_types"], row["n_samples"],
                 row["bias_nats"], row["ci_lo"], row["ci_hi"], row["bias_percent"],
                 row["miller_madow_bias_nats"], row["mm_ci_lo"], row["mm_ci_hi"]))
        assert row["bias_nats"] < 0.0 or row["ci_lo"] <= 0.0 <= row["ci_hi"], \
            "a plug-in entropy read significantly HIGH: check the exact value"
        if row["ci_hi"] < 0.0:                 # a bias this run can actually resolve
            assert abs(row["miller_madow_bias_nats"]) < abs(row["bias_nats"]), \
                "the correction did not help where the bias is resolvable"
    print("    A plug-in entropy can only read LOW, and every row above is negative")
    print("    or has an interval covering zero. The coarse rows reach zero because")
    print("    a 17-word alphabet is estimable at these sizes; the fine rows do not.")
    print("    MILLER-MADOW IS NOT FREE, and the coarse rows are where that shows.")
    n_res = sum(1 for x in rows if x["ci_hi"] < 0.0)
    n_worse = sum(1 for x in rows if x["ci_hi"] >= 0.0
                  and abs(x["miller_madow_bias_nats"]) > abs(x["bias_nats"]))
    print("    It strictly reduces |bias| on all %d rows whose bias this run can" % n_res)
    print("    RESOLVE, and on %d of the rows where the plug-in was already inside" % n_worse)
    print("    its own interval it makes the number WORSE, because it adds a")
    print("    correction to an error that was not there. Correct an entropy only")
    print("    where the cell count says there is something to correct.")
    big = [x for x in rows if x["abstraction"] == "fine"][-1]
    small = [x for x in rows if x["abstraction"] == "coarse"][-1]
    print("    READ THESE TWO. At N = %d samples the WIDE alphabet still under-reads"
          % big["n_samples"])
    print("    its exact %.6f nats by %.4f, which is %.3f%% of it; the NARROW one"
          % (big["exact_nats"], abs(big["bias_nats"]), abs(big["bias_percent"])))
    print("    under-reads its exact %.6f by %.4f, %.3f%%. A million tokens is a"
          % (small["exact_nats"], abs(small["bias_nats"]), abs(small["bias_percent"])))
    print("    realistic corpus, and it is enough for one alphabet and not the other.")
    print("    The bias is set by the CELL COUNT, not by the bed: that is the part")
    print("    that transfers to a corpus nobody can enumerate.")

    print("\n(h) T-VERB, EVERY READING ON ONE EVENT WITH ONE STATISTIC.")
    t = t_verb_report()
    print("    event: %s." % ABSTENTION)
    print("    overall abstention rate %.6f over %d occurrences, by cause:"
          % (t["overall_rate"], r["n_verb_occurrences"]))
    for code, k in sorted(t["abstention_by_cause"].items()):
        print("        %-18s %8d" % (code, k))
    assert sum(t["abstention_by_cause"].values()) == g["n_abstentions_total"]
    print("    all arms on ONE denominator, %d occurrences: the %d where some arm"
          % (t["occurrence_arms"][0]["n_occurrences"],
             t["occurrence_arms"][0]["n_dropped"]))
    print("    REFUSES a grade are dropped from every arm alike.")
    print("    %-27s %-8s %-24s %s" % ("ARM", "AUC", "Hanley-McNeil CI", "levels"))
    for a in t["occurrence_arms"]:
        print("    %-27s %.6f  [%.6f, %.6f] %6d"
              % (a["name"], a["auc"], a["ci_lo"], a["ci_hi"], a["levels"]))
    by = {a["name"]: a["auc"] for a in t["occurrence_arms"]}
    assert by["H(state 2ply | verb)"] == by["out-degree of successor"]
    print("    The second and fourth rows are EQUAL to the last bit, which is the")
    print("    whole point: AUC is a rank statistic and ln is monotone, so reading")
    print("    (b) and raw out-degree are the same predictor written twice.")

    print("\n(i) THE TYPE-LEVEL GRADE, per verb type, against three controls.")
    print("    %-14s %-9s %-26s" % ("ARM", "rho", "bootstrap CI, n = %d verbs"
                                    % t["n_verb_types"]))
    for a in t["arms"]:
        print("    %-14s %+.4f   CI [%+.4f, %+.4f]" % (a["name"], a["rho"],
                                                       a["ci_lo"], a["ci_hi"]))
    p = t["paired_vs_degree"]
    print("    PAIRED, same resamples: |rho entropy| - |rho %s| = %+.4f"
          % (p["against"], p["delta"]))
    print("    CI [%+.4f, %+.4f]  ->  %s" % (p["ci_lo"], p["ci_hi"], p["verdict"]))

    print("\n(j) THE BINNED TABLE, every rate beside its count and its interval.")
    print("    %-4s %-18s %6s %10s %9s %8s %-22s"
          % ("bin", "H range (nats)", "verbs", "occ", "events", "rate", "Wilson CI"))
    for b in t["bins"]:
        print("    %-4d [%.4f, %.4f] %6d %10d %9d %8.4f [%.4f, %.4f]"
              % (b["bin"], b["h_lo"], b["h_hi"], b["n_verbs"], b["n_occurrences"],
                 b["n_events"], b["rate"], b["ci_lo"], b["ci_hi"]))
    assert sum(b["n_occurrences"] for b in t["bins"]) == r["n_verb_occurrences"]

    print("\n(k) THE STRATA, where the two predictors are finally separated.")
    s = t["strata"]
    print("    entropy scored WITHIN deciles of out-degree : mean rho %+.4f"
          % s["mean_rho_entropy_within_degree"])
    print("    out-degree scored WITHIN deciles of entropy : mean rho %+.4f"
          % s["mean_rho_degree_within_entropy"])
    print("    signs of the ten entropy-within-degree strata: %s"
          % " ".join("%+d" % (1 if x["rho"] > 0 else -1)
                     for x in s["entropy_within_degree"]))

    print("\n(l) THE VERDICT ON T-VERB, carried in the module and not in a report.")
    print("    T-VERB said refusals concentrate at HIGH-H verbs. On this bed")
    print("    %s -- the OPPOSITE sign. The association" % t["direction"])
    print("    is real and large, rho %+.4f, but out-degree alone reads %+.4f on"
          % (t["arms"][0]["rho"], t["arms"][1]["rho"]))
    print("    the same verbs, the paired difference is %+.4f with a CI that"
          % p["delta"])
    print("    excludes zero, and at matched out-degree the entropy keeps a mean rho")
    print("    of only %+.4f while out-degree keeps %+.4f at matched entropy."
          % (s["mean_rho_entropy_within_degree"], s["mean_rho_degree_within_entropy"]))
    print("    T-VERB is STRUCK on this bed: wrong direction, degenerate in two of")
    print("    its four readings, and in the two where it is not degenerate it is")
    print("    beaten by the cheaper control it had to beat.")
    print("    THE ROUTE, because a kill needs one in the same report: the grade-2")
    print("    slot on a token should carry the successor's OUT-DEGREE. It is one")
    print("    integer per state against a %d x %d bigram matrix, so the reroute is"
          % (t["n_verb_types"], t["n_verb_types"]))
    print("    cheaper as well as better, and it needs no estimator at all. Keep H")
    print("    only where it is reported BESIDE log(degree) with the paired interval.")
    print("    WHAT SURVIVES, and it is the larger half: the estimator curve in (g).")
    print("    It does not depend on T-VERB and it prices every conditional entropy")
    print("    this programme reads off a corpus it cannot enumerate.")

    print("\n(m) PRIOR ART, and what is NOT CLAIMED. Grading a token by the")
    print("    conditional entropy of what follows it is unoccupied ground in the")
    print("    addendum's four references. What is NOT claimed here: the entropy")
    print("    itself (Shannon 1948), the plug-in bias and its correction (Miller")
    print("    1955, Basharin 1959), the AUC and its interval (Hanley and McNeil")
    print("    1982), the Wilson interval (Wilson 1927), and the enumerated bed and")
    print("    its exact oracle, which belong to ceqjepa.chess_steps and were read")
    print("    here, never edited. The conjunction is the only claim.")

    took = time.perf_counter() - t_start
    print("\n    self-check wall clock %.1f s   (budget 300 s, this box, CPU)" % took)
    print("ALL SELF-CHECKS PASSED")


if __name__ == "__main__":
    demo()
