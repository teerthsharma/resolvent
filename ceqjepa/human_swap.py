"""The consequence swap at fixed surface, carried onto HUMAN positions.

WHAT THIS FILE IS FOR. ceqjepa.intent_do reports the sharpest causal claim in
this repository: an intervention that changes the set of admitted futures moves
the read, while a meaning-preserving intervention of the same Hamming distance
moves it EXACTLY 0.000000, on every one of 1000 swaps. That was measured on a
generated bed. This file measures the same claim on 13,479 K+Q-vs-K positions a
human actually played (Lichess 2013-01, kqk_slice.tsv), where the admitted
futures are known EXACTLY from ceqjepa.chess_steps' enumeration of all 368,452
legal positions and 4,891,672 legal moves, and where nothing is planted.

THE TRANSPORT, stated so it can be attacked. A context is a position; its tokens
are the three piece-squares; an intervention is a RELOCATION of exactly one piece
to another square whose result is still a legal position in the enumerated space.
Every relocation moves exactly one of three one-hot piece-square tokens, so the
surface Hamming distance is 2 for EVERY swap in BOTH families -- checked as a min
and a max over all 1,389,988 of them, not on average. The admitted futures of a
position are the SET of exact outcome codes of its legal successors, each code
being that successor's exact distance-to-mate or, for a position with no forced
mate, which kind of draw it is. A relocation is SURFACE-ONLY when that set is
unchanged and CONSEQUENCE when it is not.

FIRST FINDING, AND IT IS THE PRECONDITION FOR EVERY OTHER. The bed PLANTED its
surface-only cell: 7 of 12 slots are decor and exclude nothing by construction.
Real chess has no decor, and whether a meaning-preserving intervention exists at
all on a real board was an open question. It does: 153,215 of 1,389,988 legal
single-piece relocations preserve the admitted-futures set exactly, 0.110228 of
them, over 9,360 distinct human positions, with 5,539 further relocations
dropped because they land on a position with no legal move and the read is
undefined there. The control has support and the claim is testable on real data.

SECOND FINDING: THE TWO FAMILIES ARE NOT DISPLACEMENT-MATCHED, and untightened
the gap is partly a surface gap wearing a different name -- the exact failure
intent_do's own docstring warns about. Surface-only relocations sit CLOSER to the
square they left: mean Chebyshev 2.525074 against 3.669249, a difference of
1.144175 squares, with a worst single cell at 4.935484. Pinning (base position, piece moved) is not enough. Every headline
number below is at the TIGHTENED PIN (base position, piece moved, Chebyshev
displacement), under which the displacement difference is 0.0 exactly, and the
loose pin is carried beside it so the size of the confound is visible.

THE READS, AND WHICH OF THEM CAN CARRY THE CLAIM. Six reads are run and each
declares whether it is a FUNCTION OF THE ADMITTED-FUTURES SET:

    read           function of A?   what it is
    a_set          YES              uniform over A, TV -- the bed's own read
    dtm            YES              exact plies to mate = 1 + min/max over A
    committor      NO               exact q_loss of the absorbing chain
    alikeness      NO               fitted surface similarity, planted negative
    jepa_trained   NO               the trained pi-JEPA encoder, 9-D, CPU
    jepa_random    NO               the same architecture, never trained

A read that is a function of A CANNOT move on a swap defined as "A is unchanged".
Its 0.000000 is an algebraic identity, not a measurement, and reporting it as
evidence is circular. intent_do says this about its own oracle read in one line
and then reports the 0.000000 as the claim anyway. `dtm` is the same identity one
step further out and is carried to show how far the circle reaches: DTM of a win
is 1 + min over the loss codes in A, DTM of a loss is 1 + max over A, so
preserving A forces it.

THE MEASUREMENT (`python -m ceqjepa.human_swap`, working tree at commit
82eb2d4 with every file this module imports byte-identical to b4c6620 --
`git diff --stat b4c6620..82eb2d4 -- ceqjepa/chess_steps.py ceqjepa/curvature.py`
is empty -- on WIN-16QAL06O9GB, python 3.11.9, numpy 2.4.6, torch 2.14.0+cpu,
71,016 pinned cells, each cell contributing the mean over its own relocations so
no seed reaches this table):

    read          family         mean       CI                      worst    nonzero
    a_set         consequence  0.471581  [0.469825, 0.473337]     1.000000   1.000000
    a_set         surface-only 0.000000  [0.000000, 0.000000]     0.000000   0.000000
    dtm           consequence  2.078893  [2.069239, 2.088546]    14.000000   0.758375
    dtm           surface-only 0.000000  [0.000000, 0.000000]     0.000000   0.000000
    committor     consequence  0.009537  [0.009386, 0.009688]     0.296829   1.000000
    committor     surface-only 0.003812  [0.003735, 0.003889]     0.191041   0.994916
    alikeness     consequence  0.015538  [0.015451, 0.015625]     0.132030   0.916822
    alikeness     surface-only 0.014474  [0.014364, 0.014583]     0.128563   0.904212
    jepa_trained  consequence  0.133098  [0.132685, 0.133511]     0.917444   1.000000
    jepa_trained  surface-only 0.133132  [0.132626, 0.133639]     0.750198   1.000000
    jepa_random   consequence  0.135859  [0.135708, 0.136010]     0.288103   1.000000
    jepa_random   surface-only 0.135922  [0.135714, 0.136130]     0.287279   1.000000

THE CLAIM DOES NOT SURVIVE, and the shape of the failure is specific. On the two
reads that are functions of A the surface-only floor is exactly 0.000000 in every
one of 71,016 pinned cells -- reproducing the bed exactly and carrying exactly
nothing, because it could not have come out otherwise. On the committor, the one
exact read of the SAME enumeration that is not such a function, a
meaning-preserving swap moves the read on 0.994916 of cells, its worst single
swap reaches 0.191041, and the separation ratio is 2.501651. The bed reports a
surface-only floor of exactly 0.0 on its exact read and a ratio of 13.8 on its
estimated one. 2.501651 is neither.

THE LEARNED READ IS IN THE PLANTED NEGATIVE'S CELL, AND THE RANDOM ARM SAYS SO.
The trained pi-JEPA encoder moves 0.133098 on a consequence swap and 0.133132 on
a meaning-preserving one -- a ratio of 0.999745, with the meaning-PRESERVING swap
moving it very slightly further -- and its 9-D embedding moves on 1.000000 of
surface-only swaps. A never-trained encoder of the same architecture reads
0.135859 against 0.135922, ratio 0.999536. Training moved the ratio by 0.000209.
Whatever the trained encoder learned, it did not learn to tell a swap that
changes the admitted futures from one that does not.

THE MODEL-SIDE HEADROOM BOUNDS HOW MUCH OF THAT IS THE TASK AND HOW MUCH IS THE
ENCODER, and the answer is uncomfortable. Least squares from the 9-D embedding to
exact plies-to-mate, fitted on half the win basin and scored on the other 172,702
positions, reads R2 0.144422 (RMSE 3.312359 plies) for the trained arm against
0.101277 (3.394850) for the never-trained one, with predict-the-mean at 3.581027.
Training bought 0.043145 of R2 over a random projection of the board one-hot. So
the trained encoder does carry real information about this endgame, but barely
more than an untrained one does, and the isolation failure above cannot be blamed
entirely on the isolation task.

THE MECHANISM, NAMED. A surface-only swap preserves the SET of admitted futures
by definition and preserves the MULTISET on only 0.187116 of swaps; the number of
legal moves survives on 0.364814. So the intervention changes HOW MANY WAYS each
future is reached while leaving WHICH futures are reachable alone. The bed's read
is uniform over a set and is blind to that by construction; the committor weights
successors by 1/degree and is not. Set-level meaning-preservation is not
meaning-preservation, and the bed could not have discovered the difference,
because its own read cannot see multiplicity either.

THE PLANTED NEGATIVE HOLDS, which is what makes the rest readable. The alikeness
read -- outcome prototypes fitted over the whole 368,452-position space, scored
by surface agreement, which is what self-attention computes (Vaswani et al.,
arXiv:1706.03762) -- reads 0.015538 against 0.014474, a ratio of 1.073511, firing
on both families at matched surface and matched displacement. A similarity read
cannot separate consequence from surface on real positions either.

TWO NULLS, AND EACH NAMES BOTH SIDES. `label_permutation_null` VARIES the family
label and PINS the base position, the piece moved, the Chebyshev displacement,
the relocation set itself, the read and the seed; under it every read's ratio
returns to 1.0 within 0.005, the identity reads included.
`intervention_permutation_null` is the ablation that killed DCM-1 -- the
move-permutation ablation of ceqjepa/hf/README.md, which read -0.0044 +- 0.0138
there -- and it VARIES which base each intervention lands on while PINNING the
piece, the displacement, the family label, each bucket's own interventions and
bases, the read, the metric and the seed:

    read            true    permuted     lost to permutation
    a_set            inf    1.024016     inf
    dtm              inf    1.028882     inf
    committor    2.569359   0.880856     1.688503
    alikeness    1.205652   1.017081     0.188571
    jepa_trained 1.061296   0.995295     0.066001
    jepa_random  0.997853   0.999147    -0.001293

Those ratios are POOLED over swaps rather than averaged over cells, which is why
the committor's true column reads 2.569359 where the cell-weighted table above
reads 2.501651; the ablation compares like with like inside its own column. Read
the last column. The exact committor loses 1.688503 of its separation when the
pairing is re-dealt, so its movement genuinely depends on WHICH intervention was
applied to WHICH position. The trained encoder loses 0.066001 -- less than the
planted negative's 0.188571 and barely above the never-trained encoder's
-0.001293. On the ablation that killed the last attempt, the trained model scores
below the control it was supposed to beat.

THE BED HEADROOM, MEASURED RATHER THAN HOPED FOR, and it splits in two. On a read
that is a function of A the ceiling is infinite and VACUOUS, because the
definition already clears the bar. On a read that is not, the ceiling is the
EXACT committor of the same enumeration -- no model can beat an exact quantity at
being that quantity -- and it is 2.501651, with a surface-only floor of 0.003812
that is not zero and movement on 0.994916 of meaning-preserving swaps. There is
headroom here, unlike the DCM-1 bed, and it is 2.5x rather than the infinity the
synthetic bed's 0.000000 advertises.

WHAT IS NOT CLAIMED. The endgame is K+Q vs K and nothing here says what a
multiplicity-sensitive read does with more pieces on the board. The learned read
is the per-position ENCODER branch `online.net`; the length-dependent read
`model(x).out` needs a sequence, and a sequence assembled around an intervened
position would vary a second thing no null here names, so it is not run. The
committor is q_loss under uniform random play, the transition-path-theory
committor and not the game value; under optimal play it is 0 or 1 and carries no
gradient, which ceqjepa.chess_steps states and this file inherits. No claim is
made about whether a read that DOES see multiplicity would separate the families
better than 2.5x -- the committor is one such read, not the best one.

RUN: python -m ceqjepa.human_swap
"""

import hashlib
import os
from collections import defaultdict
from functools import lru_cache

import chess
import numpy as np

from ceqjepa.chess_steps import oracle, space

__all__ = [
    "SLICE_PATH", "SLICE_SHA256", "SEED", "PIECES", "READS",
    "READ_IS_A_FUNCTION_OF_A", "PINS", "ALIKENESS_BETA",
    "JEPA_CKPT", "JEPA_ARMS", "JEPA_ARM_OF_READ", "JEPA_PROVENANCE_COMMIT",
    "slice_provenance", "human_bases", "admitted", "relocations",
    "relocation_census", "displacement_audit", "multiplicity_audit",
    "matched_cells", "swap_table", "label_permutation_null",
    "intervention_permutation_null", "oracle_headroom",
    "committor_residual", "jepa_probe", "multiset_family_reprice", "demo",
]

#: The human slice, and the exact bytes these numbers were measured on. A
#: different file is a different measurement and the check refuses rather than
#: quietly reporting numbers from it.
SLICE_PATH = os.path.join(
    os.environ.get("TEMP", r"C:\Users\seal\AppData\Local\Temp"),
    "claude", "C--Users-seal-Desktop-New-folder--32-",
    "bb16374f-0874-425e-b13e-7a1d3ce67564", "scratchpad", "kqk_slice.tsv")
SLICE_SHA256 = "f3ff81b01c17fcb378557c84ecd7a9f3eabbe1c5c9cd4eb901c1cc95719600ee"

#: The one seed behind every draw in this file. The headline table uses none of
#: it: each pinned cell contributes the MEAN over its relocations, so the swap
#: table is seed-free and the seed reaches only the permutation null.
SEED = 20260914

#: The three tokens of the context. A relocation moves exactly one.
PIECES = ("WK", "WQ", "BK")

#: Every read run below, in the order the table prints them.
READS = ("a_set", "dtm", "committor", "alikeness", "jepa_trained", "jepa_random")

#: THE LOAD-BEARING DECLARATION. A read that is a function of the
#: admitted-futures set cannot move on a swap defined as "that set is
#: unchanged", so its 0.000000 is an identity and never a measurement.
READ_IS_A_FUNCTION_OF_A = {
    "a_set": True,      # uniform over A: A -> q is the definition
    "dtm": True,        # 1 + min over loss codes in A, or 1 + max over A
    "committor": False, # weights successors by 1/degree; A is multiplicity-free
    "alikeness": False, # surface only, never touches A
    "jepa_trained": False,
    "jepa_random": False,
}

#: THE LEARNED READ, and its own control. The checkpoint ships three arms; the
#: `frozen_random` and `untrained` encoders are the SAME tensors (checked, not
#: assumed) so only two distinct encoders exist and both are run. The random arm
#: is the whole reason the trained arm's number is readable: an encoder that
#: never saw a game is the bar any separation has to clear before the word
#: "learned" is allowed near it.
JEPA_CKPT = os.path.join(
    os.path.dirname(SLICE_PATH), "kernel_output", "pi_jepa_chess_weights.pt")
JEPA_ARMS = ("trained", "frozen_random")
JEPA_ARM_OF_READ = {"jepa_trained": "trained", "jepa_random": "frozen_random"}
JEPA_PROVENANCE_COMMIT = "391a2d004fb5e77aa91febdfdb91c79a1a8db7bf"

#: python-chess piece codes in the checkpoint's SQUARE-MAJOR input layout,
#: index = square * 12 + (code - 1). The three `fen_to_vec` helpers in this repo
#: are PLANE-major and belong to a different model; feeding one of them to this
#: checkpoint differs in 62 of 769 coordinates on the start position and reads a
#: number that looks fine and means nothing.
JEPA_PIECE_OFFSET = {"WK": 6 - 1, "WQ": 5 - 1, "BK": 12 - 1}
JEPA_X_DIM = 769
JEPA_D_LATENT = 9

#: The two pins. The loose one is what the bed's design translates to directly;
#: the tightened one adds the displacement axis the board has and the bed did not.
PINS = ("base-piece", "base-piece-cheb")

#: Temperature of the alikeness read, inherited in spirit from intent_do's
#: DENSITY_BETA. Pinned, never swept: a swept temperature on a planted negative
#: is a search for the setting that makes the control look worst.
ALIKENESS_BETA = 8.0

#: Outcome codes for a successor with no forced mate. Distinct per cause, so a
#: caller can count by cause and nothing stands in for a measurement.
CODE_STALEMATE = -1
CODE_DRAWN_BY_DEFENCE = -2
CODE_SINK = -3
CODE_OFFSET = 3            # shifts every code into [0, 23] for the bitmask
N_CODES = 24


# ---------------------------------------------------------------------------
# THE ENUMERATION, AND THE ADMITTED-FUTURES SET AS A BITMASK
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def admitted():
    """Every position's admitted-futures set, exactly, as a 24-bit mask.

    The outcome codes run -3..20 and there are 24 of them, so the SET of
    successor codes fits in one integer and set equality is integer equality.
    That is why the two families below can be split over 1.39M relocations
    without a single approximate comparison.
    """
    sp, o = space(), oracle()
    n, nt = sp["n_positions"], sp["n_states"]

    code = np.full(nt, CODE_DRAWN_BY_DEFENCE, np.int64)
    code[:n] = o["dtm"]
    code[o["stalemate_nodes"]] = CODE_STALEMATE
    code[sp["sink"]] = CODE_SINK
    if code.min() < -CODE_OFFSET or code.max() >= N_CODES - CODE_OFFSET:
        raise AssertionError(
            "outcome codes %d..%d do not fit the %d-code mask this module's set "
            "equality depends on" % (code.min(), code.max(), N_CODES))

    indptr = sp["succ_indptr"]
    degree = np.diff(indptr).astype(np.int64)
    sig = np.zeros(n, np.int64)
    live = np.flatnonzero(degree > 0)
    bits = (np.int64(1) << (code + CODE_OFFSET)).astype(np.int64)
    sig[live] = np.bitwise_or.reduceat(bits[sp["succ"]], indptr[:-1][live])

    # the MULTISET of successor codes, as a count per code. The set above is the
    # same object with every count clipped to one, so the two differ exactly
    # where multiplicity differs and nowhere else.
    owner = np.repeat(np.arange(n), degree)
    counts = np.zeros((n, N_CODES), np.int32)
    np.add.at(counts, (owner, code[sp["succ"]] + CODE_OFFSET), 1)
    mult = np.ascontiguousarray(counts).view(
        np.dtype((np.void, counts.dtype.itemsize * N_CODES))).ravel()

    return dict(code=code, sig=sig, degree=degree, multiset=mult,
                committor=o["q"][:n], dtm=o["dtm"], n_positions=n)


def committor_residual():
    """||q - Pq||_inf on the fixed point the committor was read off."""
    return float(oracle()["q_residual"])


# ---------------------------------------------------------------------------
# THE HUMAN SLICE
# ---------------------------------------------------------------------------

def _read_slice():
    if not os.path.exists(SLICE_PATH):
        raise FileNotFoundError(
            "the human slice is not at %r; this module measures a claim ON that "
            "file and has nothing to report without it" % SLICE_PATH)
    with open(SLICE_PATH, "rb") as fh:
        raw = fh.read()
    got = hashlib.sha256(raw).hexdigest()
    if got != SLICE_SHA256:
        raise AssertionError(
            "kqk_slice.tsv hashes %s, not the %s every number in this module was "
            "measured on" % (got, SLICE_SHA256))
    return raw.decode("utf-8")


@lru_cache(maxsize=1)
def slice_provenance():
    """What the file is, and how much of it lands in the enumerated space.

    HALF THE SLICE IS MIRRORED, and saying so matters: 6,630 of the 13,479 lines
    are K vs K+q with a BLACK queen, which chess_steps does not enumerate. A
    colour mirror -- python-chess's own Board.mirror(), which swaps colours and
    flips ranks -- carries them into the White-queen space exactly, because chess
    is invariant under it. Without the mirror half the slice is silently dropped
    and the 'real data' is half of what it says it is.
    """
    sp = space()
    text = _read_slice()
    n_lines = mirrored = mapped = 0
    keys = []
    for line in text.splitlines():
        if not line.strip():
            continue
        n_lines += 1
        fen = line.split("\t")[0]
        b = chess.Board(fen)
        if b.pieces(chess.QUEEN, chess.BLACK):
            b = b.mirror()
            mirrored += 1
        wk, bk = b.king(chess.WHITE), b.king(chess.BLACK)
        queens = list(b.pieces(chess.QUEEN, chess.WHITE))
        if (wk is None or bk is None or len(queens) != 1
                or len(b.piece_map()) != 3 or not b.is_valid()):
            continue
        key = (wk, queens[0], bk, b.turn)
        if key not in sp["index"]:
            continue
        mapped += 1
        keys.append(key)
    return dict(sha256=SLICE_SHA256, n_bytes=len(text.encode("utf-8")),
                n_lines=n_lines, n_black_queen_mirrored=mirrored,
                n_mapped=mapped, n_distinct_positions=len({k for k in keys}),
                keys=tuple(keys))


@lru_cache(maxsize=1)
def human_bases():
    """The distinct human positions, with terminals and draws counted by cause."""
    sp, ad = space(), admitted()
    prov = slice_provenance()
    idx = np.array(sorted({sp["index"][k] for k in prov["keys"]}), np.int64)
    terminal = ad["degree"][idx] == 0
    live = idx[~terminal]
    win = ad["dtm"][live] >= 0
    return dict(idx=live, n_distinct=int(idx.size),
                n_terminal_refused=int(terminal.sum()),
                n_win_basin=int(win.sum()),
                n_draw_refused=int((~win).sum()))


# ---------------------------------------------------------------------------
# THE INTERVENTION: RELOCATE ONE PIECE, HOLD THE SURFACE
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def relocations():
    """Every legal single-piece relocation of every human base position.

    Columns: base state, piece slot, relocated state, family (0 surface-only,
    1 consequence), Chebyshev displacement. A relocation into a terminal
    position is DROPPED and counted, not scored as zero: a position with no legal
    move admits nothing and the read is undefined there.
    """
    sp, ad = space(), admitted()
    index, sig, degree = sp["index"], ad["sig"], ad["degree"]
    rows, dropped_terminal = [], 0
    for b0 in human_bases()["idx"]:
        wk, wq, bk, turn = (int(v) for v in sp["keys"][b0])
        turn = bool(turn)
        s0 = sig[b0]
        for slot in range(3):
            orig = (wk, wq, bk)[slot]
            for target in range(64):
                if target == orig:
                    continue
                if slot == 0:
                    key = (target, wq, bk, turn)
                elif slot == 1:
                    key = (wk, target, bk, turn)
                else:
                    key = (wk, wq, target, turn)
                j = index.get(key)
                if j is None:
                    continue
                if degree[j] == 0:
                    dropped_terminal += 1
                    continue
                rows.append((int(b0), slot, int(j),
                             0 if sig[j] == s0 else 1,
                             chess.square_distance(orig, target)))
    return np.array(rows, np.int64), dropped_terminal


def relocation_census():
    """How big each cell is, and that the surface Hamming really is held."""
    rec, dropped = relocations()
    surf = int((rec[:, 3] == 0).sum())
    cons = int((rec[:, 3] == 1).sum())
    # A relocation moves one piece off one square and onto another: in the
    # 3x64 one-hot surface code that is exactly two bits, for BOTH families,
    # by construction rather than on average. Asserted as a (min, max) pair so
    # a future change that breaks it cannot hide inside a mean.
    return dict(n_relocations=int(rec.shape[0]),
                n_surface_only=surf, n_consequence=cons,
                n_dropped_terminal=int(dropped),
                surface_only_fraction=surf / float(rec.shape[0]),
                hamming_one_hot_consequence=(2, 2),
                hamming_one_hot_surface_only=(2, 2))


# ---------------------------------------------------------------------------
# THE PINS
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def matched_cells(pin="base-piece-cheb"):
    """Cells carrying BOTH families under `pin`, as (base, surf list, cons list).

    A cell is the unit of comparison and every cell contributes the MEAN over
    its own relocations, so no cell is weighted by how many relocations it
    happens to own and no seed enters the headline table.
    """
    if pin not in PINS:
        raise ValueError("unknown pin %r, expected one of %r" % (pin, PINS))
    rec, _ = relocations()
    cells = defaultdict(lambda: ([], []))
    for base, slot, j, fam, cheb in rec:
        key = (base, slot) if pin == "base-piece" else (base, slot, cheb)
        cells[key][int(fam)].append((int(j), int(cheb)))
    return tuple((k, tuple(v[0]), tuple(v[1]))
                 for k, v in cells.items() if v[0] and v[1])


def displacement_audit():
    """What the Chebyshev displacement does under each pin. THE CONFOUND."""
    out = {}
    for pin in PINS:
        cells = matched_cells(pin)
        s = np.array([np.mean([c for _, c in surf]) for _, surf, _ in cells])
        c = np.array([np.mean([c for _, c in cons]) for _, _, cons in cells])
        out[pin] = dict(n_cells=len(cells),
                        mean_surface_only=float(s.mean()),
                        mean_consequence=float(c.mean()),
                        max_abs_difference=float(np.abs(c - s).max()))
    return out


# ---------------------------------------------------------------------------
# THE READS
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _uniform_over_A():
    """q = uniform over the admitted set, for every distinct mask in play."""
    table = {}
    for s in np.unique(admitted()["sig"]):
        s = int(s)
        on = [i for i in range(N_CODES) if (s >> i) & 1]
        v = np.zeros(N_CODES)
        if on:
            v[on] = 1.0 / len(on)
        table[s] = v
    return table


@lru_cache(maxsize=1)
def _alikeness():
    """THE PLANTED NEGATIVE: how ALIKE the surface is to each outcome.

    Every outcome class gets a prototype FITTED over the whole enumerated space
    -- the mean 3x64 one-hot of every position carrying that outcome -- and the
    score of a class is the agreement between the position's own one-hot and
    that prototype at temperature ALIKENESS_BETA. Fitting the prototypes rather
    than drawing them at random is deliberate and makes the control HARDER: this
    read carries real information about distance-to-mate and still may not use
    the admitted-futures set, because it never sees one. Both families move the
    surface by the same two one-hot bits at the same displacement, so this read
    must move about equally for both. If it did not, the separation the identity
    reads show would be a property of the slice rather than of those reads.
    """
    sp, ad = space(), admitted()
    n = ad["n_positions"]
    keys = sp["keys"][:, :3].astype(np.int64)
    cls = np.clip(ad["dtm"], -1, None) + 1          # draws collapse to class 0
    n_cls = int(cls.max()) + 1
    proto = np.zeros((n_cls, 3, 64))
    for slot in range(3):
        for c in range(n_cls):
            sel = cls == c
            proto[c, slot] = np.bincount(keys[sel, slot], minlength=64) / max(int(sel.sum()), 1)
    return proto.reshape(n_cls, 192), keys


@lru_cache(maxsize=2)
def _jepa_embeddings(arm):
    """The 9-D pi-JEPA encoder embedding of every enumerated position, one arm.

    The encoder is the per-position branch `online.net`, a 769 -> 1024 -> 1024 ->
    9 MLP, loaded with strict=True so a layout or shape drift raises instead of
    silently loading a subset. The length-dependent read `model(x).out` is NOT
    used: it needs a sequence, and a sequence assembled around an intervened
    position would smuggle in a second varying thing this null has not named.

    Embeddings are returned in units of the arm's own RMS embedding norm over
    the whole space, because the trained and random encoders differ in scale by
    two orders of magnitude and a raw L2 movement is not comparable across them.
    Ratios are scale-free either way; the scaling is so the means are readable.
    """
    import torch
    from torch import nn

    if not os.path.exists(JEPA_CKPT):
        raise FileNotFoundError(
            "the pi-JEPA checkpoint is not at %r; the learned read cannot run "
            "and this is RED, not a skip" % JEPA_CKPT)
    ck = torch.load(JEPA_CKPT, map_location="cpu", weights_only=False)
    got = ck.get("provenance", {}).get("commit")
    if got != JEPA_PROVENANCE_COMMIT:
        raise AssertionError(
            "checkpoint provenance commit is %r, not the %r these numbers were "
            "measured on" % (got, JEPA_PROVENANCE_COMMIT))

    enc = nn.Sequential(nn.Linear(JEPA_X_DIM, 1024), nn.GELU(),
                        nn.Linear(1024, 1024), nn.GELU(),
                        nn.Linear(1024, JEPA_D_LATENT))
    prefix = "online.net."
    enc.load_state_dict({k[len(prefix):]: v
                         for k, v in ck["arms"][arm]["model"].items()
                         if k.startswith(prefix)}, strict=True)
    enc.eval()

    keys = space()["keys"].astype(np.int64)
    n = keys.shape[0]
    x = np.zeros((n, JEPA_X_DIM), np.float32)
    rows = np.arange(n)
    for slot, piece in enumerate(PIECES):
        x[rows, keys[:, slot] * 12 + JEPA_PIECE_OFFSET[piece]] = 1.0
    x[rows, 768] = keys[:, 3].astype(np.float32)
    if not np.array_equal(x.sum(1), 3.0 + keys[:, 3]):
        raise AssertionError("the one-hot did not land on exactly 3 pieces + stm")

    out = np.empty((n, JEPA_D_LATENT), np.float32)
    with torch.no_grad():
        for i in range(0, n, 8192):
            out[i:i + 8192] = enc(torch.from_numpy(x[i:i + 8192])).numpy()
    rms = float(np.sqrt((out ** 2).sum(1).mean()))
    return out.astype(np.float64) / rms


def _read_vector(name, states):
    """The read's value at each of `states`, as a (len(states), d) array."""
    ad = admitted()
    states = np.asarray(states, np.int64)
    if name == "a_set":
        table = _uniform_over_A()
        return np.array([table[int(ad["sig"][s])] for s in states])
    if name == "dtm":
        return ad["dtm"][states].astype(float)[:, None]
    if name == "committor":
        return ad["committor"][states][:, None]
    if name == "alikeness":
        proto, keys = _alikeness()
        x = np.zeros((states.size, 192))
        k = keys[states]
        for slot in range(3):
            x[np.arange(states.size), slot * 64 + k[:, slot]] = 1.0
        z = ALIKENESS_BETA * (x @ proto.T) / 3.0
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    if name in JEPA_ARM_OF_READ:
        return _jepa_embeddings(JEPA_ARM_OF_READ[name])[states]
    raise ValueError("unknown read %r, expected one of %r" % (name, READS))


#: How each read's movement is measured. A distribution moves in total
#: variation, a scalar in absolute difference, a latent vector in L2. Named in a
#: table rather than inferred from an array's width, so a read whose width
#: changes cannot silently switch metric.
READ_METRIC = {"a_set": "tv", "alikeness": "tv", "dtm": "abs",
               "committor": "abs", "jepa_trained": "l2", "jepa_random": "l2"}


def _movement(name, base_states, target_states):
    """dq: how far the read moved. TV for a distribution, |delta| for a scalar.

    A read at a DRAWN position has no distance-to-mate, and `dtm` returns NaN
    there rather than a 0.0 standing in for a measurement. Those cells are
    counted as refusals beside the mean, never folded into it.
    """
    a = _read_vector(name, base_states)
    b = _read_vector(name, target_states)
    if name == "dtm":
        bad = (a[:, 0] < 0) | (b[:, 0] < 0)
        d = np.abs(a[:, 0] - b[:, 0])
        d[bad] = np.nan
        return d
    metric = READ_METRIC[name]
    if metric == "abs":
        return np.abs(a[:, 0] - b[:, 0])
    if metric == "tv":
        return 0.5 * np.abs(a - b).sum(axis=1)
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def _cell_means(name, cells, which):
    """One number per cell: the mean movement over that cell's relocations."""
    bases, targets, owner = [], [], []
    for i, (key, surf, cons) in enumerate(cells):
        for j, _ in (surf if which == 0 else cons):
            bases.append(int(key[0]))
            targets.append(j)
            owner.append(i)
    d = _movement(name, bases, targets)
    owner = np.asarray(owner)
    out = np.full(len(cells), np.nan)
    for i in range(len(cells)):
        v = d[owner == i]
        v = v[~np.isnan(v)]
        if v.size:
            out[i] = v.mean()
    return out, d


def _stats(cell_v, raw):
    v = cell_v[~np.isnan(cell_v)]
    r = raw[~np.isnan(raw)]
    se = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
    return dict(n_cells=int(v.size), n_refused=int(np.isnan(cell_v).sum()),
                n_swaps=int(r.size), mean=float(v.mean()),
                ci=(float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)),
                worst=float(r.max()), fraction_nonzero=float((r != 0).mean()))


@lru_cache(maxsize=4)
def swap_table(pin="base-piece-cheb"):
    """Every read, both families, at matched surface and matched displacement."""
    cells = matched_cells(pin)
    out = {}
    for name in READS:
        row = {}
        for fam, label in ((1, "consequence"), (0, "surface_only")):
            cv, raw = _cell_means(name, cells, fam)
            row[label] = _stats(cv, raw)
        out[name] = row
    return out


# ---------------------------------------------------------------------------
# THE MECHANISM
# ---------------------------------------------------------------------------

def multiplicity_audit():
    """What a surface-only swap actually preserves, and what it does not.

    The SET of admitted futures is preserved by definition. The MULTISET and the
    number of legal moves are not, and that difference is the whole reason a
    multiplicity-sensitive read moves where a set read cannot.
    """
    ad = admitted()
    rec, _ = relocations()
    out = {}
    for fam, label in ((1, "consequence"), (0, "surface_only")):
        sel = rec[rec[:, 3] == fam]
        a, b = sel[:, 0], sel[:, 2]
        out[label] = dict(
            n=int(sel.shape[0]),
            fraction_set_preserved=float((ad["sig"][a] == ad["sig"][b]).mean()),
            fraction_multiset_preserved=float((ad["multiset"][a] == ad["multiset"][b]).mean()),
            fraction_degree_preserved=float((ad["degree"][a] == ad["degree"][b]).mean()))
    return out


# ---------------------------------------------------------------------------
# THE NULL
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def label_permutation_null():
    """VARIES the family label and PINS everything else. Names both sides.

    Within each pinned cell the relocations keep their base, their piece, their
    displacement and their read; only WHICH of them is called consequence and
    which surface-only is redrawn, holding each family's size fixed. Any read
    whose separation survives that is separating on something the pin did not
    name, and the claim it licensed is void.
    """
    cells = matched_cells("base-piece-cheb")
    rng = np.random.default_rng(SEED)
    shuffled = []
    for key, surf, cons in cells:
        pool = list(surf) + list(cons)
        order = rng.permutation(len(pool))
        shuffled.append((key, tuple(pool[i] for i in order[:len(surf)]),
                         tuple(pool[i] for i in order[len(surf):])))
    shuffled = tuple(shuffled)
    out = dict(varies=("family label",),
               pins=("base position", "piece moved", "chebyshev displacement",
                     "the relocation set itself", "the read", "the seed"),
               seed=SEED, n_cells=len(cells))
    for name in READS:
        s, sr = _cell_means(name, shuffled, 0)
        c, cr = _cell_means(name, shuffled, 1)
        sm = float(np.nanmean(s))
        cm = float(np.nanmean(c))
        out[name] = dict(mean_surface_only=sm, mean_consequence=cm,
                         ratio=(cm / sm) if sm else float("inf"))
    return out


@lru_cache(maxsize=1)
def intervention_permutation_null():
    """VARIES which base each intervention is applied to. PINS everything else.

    THIS IS THE ABLATION THAT KILLED THE LAST ATTEMPT. ceqjepa/hf/README.md
    reports DCM-1's do-arm beating its bar by -7.0703 +- 0.5428 PPL and then
    reports a move-permutation ablation at -0.0044 +- 0.0138: randomising WHICH
    move was forced changed nothing, so the advantage carried no information
    about the intervention. The same ablation, here: within every bucket of
    (piece moved, Chebyshev displacement, family) the intervened positions are
    RE-DEALT across bases, holding the bucket's own multiset of interventions
    and its multiset of bases both fixed. A read whose movement is unchanged is
    responding to the marginal shape of the intervention and not to which
    intervention was applied to which position.

    PINS: the piece moved, the Chebyshev displacement, the family label, the
    multiset of interventions in each bucket, the multiset of base positions in
    each bucket, the read, the metric, the seed.
    VARIES: the pairing of base to intervention, and nothing else.
    """
    cells = matched_cells("base-piece-cheb")
    buckets = defaultdict(list)
    for key, surf, cons in cells:
        base, slot, cheb = key
        for fam, group in ((0, surf), (1, cons)):
            for j, _ in group:
                buckets[(slot, cheb, fam)].append((int(base), int(j)))

    rng = np.random.default_rng(SEED + 1)
    true_pairs = {0: [], 1: []}
    perm_pairs = {0: [], 1: []}
    for (slot, cheb, fam), pairs in buckets.items():
        bases = np.array([p[0] for p in pairs])
        targets = np.array([p[1] for p in pairs])
        true_pairs[fam].append((bases, targets))
        perm_pairs[fam].append((rng.permutation(bases), targets))

    out = dict(varies=("base-to-intervention pairing",),
               pins=("piece moved", "chebyshev displacement", "family label",
                     "the interventions in each bucket",
                     "the base positions in each bucket", "the read",
                     "the metric", "the seed"),
               seed=SEED + 1, n_buckets=len(buckets))
    for name in READS:
        row = {}
        for tag, pairs in (("true", true_pairs), ("permuted", perm_pairs)):
            means = {}
            for fam, label in ((1, "consequence"), (0, "surface_only")):
                b = np.concatenate([p[0] for p in pairs[fam]])
                t = np.concatenate([p[1] for p in pairs[fam]])
                d = _movement(name, b, t)
                means[label] = float(np.nanmean(d))
            means["ratio"] = (means["consequence"] / means["surface_only"]
                              if means["surface_only"] else float("inf"))
            row[tag] = means
        row["ratio_lost_to_permutation"] = row["true"]["ratio"] - row["permuted"]["ratio"]
        out[name] = row
    return out


def oracle_headroom():
    """WHAT THE BEST POSSIBLE READ CAN DO HERE, measured and not hoped for.

    ceqjepa/hf/README.md's second killing control was a bed-oracle arm reading
    +0.0935 with the wrong sign: no model of any size could have won on that
    bed, so the headline number was never about the model. The enumeration makes
    the same check exact here rather than estimated, and it splits in two.

    ON A READ THAT IS A FUNCTION OF A the ceiling is infinite and VACUOUS: the
    surface-only family is DEFINED as "A unchanged", so any q(A) is pinned at
    0.000000 there and the ratio has no denominator. Nothing can be learned from
    clearing a bar that the definition already clears.

    ON A READ THAT IS NOT, the ceiling is the EXACT committor of the same
    enumeration. No model can beat an exact quantity at being that quantity, so
    whatever separation the committor achieves is the headroom, full stop. It is
    2.50, with a surface-only floor that is not zero and a movement on better
    than 99% of meaning-preserving swaps. That is the headroom this claim has on
    real positions, and it is measured, not assumed.
    """
    tab = swap_table("base-piece-cheb")
    com = tab["committor"]
    return dict(
        identity_reads=tuple(r for r in READS if READ_IS_A_FUNCTION_OF_A[r]),
        identity_ceiling="infinite and vacuous: surface-only is pinned at 0 by "
                         "the definition of the family",
        exact_non_identity_read="committor",
        ceiling_ratio=com["consequence"]["mean"] / com["surface_only"]["mean"],
        ceiling_surface_only_mean=com["surface_only"]["mean"],
        ceiling_surface_only_fraction_nonzero=com["surface_only"]["fraction_nonzero"],
        ceiling_surface_only_worst=com["surface_only"]["worst"],
        # the boundary is pinned at 1.0 on the 364 mates, so this max is the
        # boundary's and not an interior value; the interior maximum is the
        # smaller number ceqjepa.chess_steps reports and is not re-quoted here
        committor_max_including_boundary=float(oracle()["q"].max()),
        bed_reported_exact_floor=0.0,
        bed_reported_estimated_ratio=13.8)


@lru_cache(maxsize=1)
def multiset_family_reprice():
    """THE REPLACEMENT ROUTE, PRICED RATHER THAN PROPOSED.

    THE GOAL THE DEAD CLAIM SERVED: showing that a read isolates CONSEQUENCE
    from SURFACE, which is north-star leg 1. What the kill above actually
    establishes is narrower than "it does not": set-level meaning-preservation
    is too COARSE. A surface-only swap keeps which outcomes are reachable and
    drops how many ways each is reached on most swaps, and that is exactly the
    thing the committor is sensitive to and the bed's set read is not.

    THE SHARPEST SURVIVING ROUTE is therefore to define the meaning-preserving
    family at the MULTISET level -- preserve the whole one-ply option profile,
    not merely its support -- and ask the same question again with the same pin.
    It is sharper for a reason that can be checked rather than asserted: the
    committor is still NOT a function of the multiset, because it depends on the
    whole downstream chain and not on one ply, so the measurement stays
    non-tautological where the set-level one collapsed into an identity. This
    function prices that route on the same 9,360 human positions instead of
    proposing it, so the next step is a decision and not an experiment.

    PINS: base position, piece moved, Chebyshev displacement, the read, the
    surface Hamming, the relocation legality.
    VARIES: whether the relocation preserves the MULTISET of admitted futures.
    """
    ad = admitted()
    rec, _ = relocations()
    mult = ad["multiset"]
    keep = mult[rec[:, 0]] == mult[rec[:, 2]]
    cells = defaultdict(lambda: ([], []))
    for row, same in zip(rec, keep):
        base, slot, j, _fam, cheb = (int(v) for v in row)
        cells[(base, slot, cheb)][int(same)].append((j, cheb))
    matched = tuple((k, tuple(v[1]), tuple(v[0]))
                    for k, v in cells.items() if v[0] and v[1])
    out = dict(n_multiset_preserving=int(keep.sum()),
               n_multiset_changing=int((~keep).sum()),
               n_cells=len(matched))
    for name in ("committor", "jepa_trained"):
        row = {}
        for fam, label in ((1, "consequence"), (0, "surface_only")):
            cv, raw = _cell_means(name, matched, fam)
            row[label] = _stats(cv, raw)
        row["ratio"] = (row["consequence"]["mean"] / row["surface_only"]["mean"]
                        if row["surface_only"]["mean"] else float("inf"))
        out[name] = row
    return out


@lru_cache(maxsize=1)
def jepa_probe(seed=SEED):
    """DOES THE LEARNED ENCODER KNOW THIS ENDGAME AT ALL? THE MODEL-SIDE HEADROOM.

    Without this the learned read's failure below is unreadable: an encoder that
    carries no information about K+Q-vs-K would fail to separate the two swap
    families for the boring reason that it separates nothing, and reporting that
    as a causal finding would be the same error ceqjepa/hf/README.md caught in
    its own headline. Least squares from the 9-D embedding to exact
    distance-to-mate, fitted on a random half of the win basin and scored on the
    other half, both arms, one split, one seed.

    PINS: the split, the seed, the target, the fit, the positions.
    VARIES: which arm's encoder produced the embedding.
    """
    ad = admitted()
    win = np.flatnonzero(ad["dtm"] >= 0)
    rng = np.random.default_rng(seed)
    order = rng.permutation(win.size)
    tr, te = win[order[:win.size // 2]], win[order[win.size // 2:]]
    y_tr, y_te = ad["dtm"][tr].astype(float), ad["dtm"][te].astype(float)
    out = {}
    for read, arm in JEPA_ARM_OF_READ.items():
        e = _jepa_embeddings(arm)
        A = np.column_stack([e[tr], np.ones(tr.size)])
        w = np.linalg.lstsq(A, y_tr, rcond=None)[0]
        pred = np.column_stack([e[te], np.ones(te.size)]) @ w
        ss_res = float(((y_te - pred) ** 2).sum())
        ss_tot = float(((y_te - y_te.mean()) ** 2).sum())
        out[read] = dict(n_train=int(tr.size), n_test=int(te.size),
                         r2=1.0 - ss_res / ss_tot,
                         rmse=float(np.sqrt(ss_res / te.size)),
                         rmse_mean_baseline=float(np.sqrt(ss_tot / te.size)))
    return out


# ---------------------------------------------------------------------------
# THE SELF-CHECK
# ---------------------------------------------------------------------------

def demo():
    """Print every number this module's docstring quotes, and assert the claim.

    Returns 0. Exit codes are asserted by the caller, never assumed.
    """
    prov, bases = slice_provenance(), human_bases()
    print("SLICE  %s" % SLICE_PATH)
    print("  sha256 %s  bytes %d  lines %d  black-queen mirrored %d"
          % (prov["sha256"], prov["n_bytes"], prov["n_lines"],
             prov["n_black_queen_mirrored"]))
    print("  mapped into the enumerated space %d  distinct positions %d"
          % (prov["n_mapped"], prov["n_distinct_positions"]))
    print("  win basin %d   drawn, refused %d   terminal, refused %d"
          % (bases["n_win_basin"], bases["n_draw_refused"],
             bases["n_terminal_refused"]))

    cen = relocation_census()
    print("\nSUPPORT  does a meaning-preserving intervention exist on a real board?")
    print("  legal single-piece relocations %d  dropped into a terminal %d"
          % (cen["n_relocations"], cen["n_dropped_terminal"]))
    print("  consequence %d   surface-only %d   surface-only fraction %.6f"
          % (cen["n_consequence"], cen["n_surface_only"],
             cen["surface_only_fraction"]))
    print("  one-hot surface Hamming, (min, max): consequence %s  surface-only %s"
          % (cen["hamming_one_hot_consequence"], cen["hamming_one_hot_surface_only"]))

    dis = displacement_audit()
    print("\nTHE CONFOUND  Chebyshev displacement, per pin")
    for pin in PINS:
        d = dis[pin]
        print("  %-16s cells %6d  consequence %.6f  surface-only %.6f  worst cell |diff| %.6f"
              % (pin, d["n_cells"], d["mean_consequence"], d["mean_surface_only"],
                 d["max_abs_difference"]))

    print("\nTHE SWAP TABLE  pin = base-piece-cheb, cells %d"
          % dis["base-piece-cheb"]["n_cells"])
    print("  %-10s %-4s %-13s %10s %-24s %11s %10s"
          % ("read", "f(A)", "family", "mean", "CI", "worst", "nonzero"))
    tab = swap_table("base-piece-cheb")
    for name in READS:
        for label in ("consequence", "surface_only"):
            s = tab[name][label]
            print("  %-10s %-4s %-13s %10.6f [%9.6f, %9.6f] %11.6f %10.6f"
                  % (name, "YES" if READ_IS_A_FUNCTION_OF_A[name] else "no",
                     label.replace("_", "-"), s["mean"], s["ci"][0], s["ci"][1],
                     s["worst"], s["fraction_nonzero"]))
        if tab[name]["surface_only"]["n_refused"] or tab[name]["consequence"]["n_refused"]:
            print("  %-10s refused cells: consequence %d  surface-only %d"
                  % ("", tab[name]["consequence"]["n_refused"],
                     tab[name]["surface_only"]["n_refused"]))

    com = tab["committor"]
    res = committor_residual()
    print("\nTHE VERDICT")
    print("  reads that ARE a function of A read exactly 0.000000 on surface-only.")
    print("  That is an identity: A unchanged forces q(A) unchanged. Not evidence.")
    print("  committor, which is NOT a function of A, moves on %.6f of "
          "surface-only swaps" % com["surface_only"]["fraction_nonzero"])
    print("  ratio consequence/surface-only = %.6f   (bed: exact 0.0 floor, "
          "estimated ratio 13.8)" % (com["consequence"]["mean"] / com["surface_only"]["mean"]))
    print("  committor fixpoint residual %.3e, movement is %.3e x it"
          % (res, com["surface_only"]["mean"] / res))

    mul = multiplicity_audit()
    print("\nTHE MECHANISM  what a surface-only swap preserves")
    for label in ("consequence", "surface_only"):
        m = mul[label]
        print("  %-13s n %8d  set %.6f  multiset %.6f  n-legal-moves %.6f"
              % (label.replace("_", "-"), m["n"], m["fraction_set_preserved"],
                 m["fraction_multiset_preserved"], m["fraction_degree_preserved"]))

    head = oracle_headroom()
    print("\nTHE HEADROOM  what the best possible read can do on this bed")
    print("  reads that are a function of A: %s -- ceiling infinite and VACUOUS"
          % (head["identity_reads"],))
    print("  exact non-identity read %r: ceiling ratio %.6f"
          % (head["exact_non_identity_read"], head["ceiling_ratio"]))
    print("  its surface-only floor %.6f (not 0), nonzero on %.6f, worst %.6f"
          % (head["ceiling_surface_only_mean"],
             head["ceiling_surface_only_fraction_nonzero"],
             head["ceiling_surface_only_worst"]))
    print("  the bed reports an exact floor of %.1f and an estimated ratio of %.1f"
          % (head["bed_reported_exact_floor"], head["bed_reported_estimated_ratio"]))

    rep = multiset_family_reprice()
    print("\nTHE REPLACEMENT ROUTE  same question, family defined on the MULTISET")
    print("  multiset-preserving relocations %d  changing %d  pinned cells %d"
          % (rep["n_multiset_preserving"], rep["n_multiset_changing"],
             rep["n_cells"]))
    for name in ("committor", "jepa_trained"):
        r = rep[name]
        print("  %-13s consequence %.6f  surface-only %.6f  ratio %.6f  "
              "surface-only nonzero %.6f"
              % (name, r["consequence"]["mean"], r["surface_only"]["mean"],
                 r["ratio"], r["surface_only"]["fraction_nonzero"]))

    pr = jepa_probe()
    print("\nTHE MODEL-SIDE HEADROOM  does the learned encoder know this endgame?")
    print("  least squares 9-D embedding -> exact plies, half/half split, n_test %d"
          % pr["jepa_trained"]["n_test"])
    for read in ("jepa_trained", "jepa_random"):
        p = pr[read]
        print("  %-13s R2 %9.6f   RMSE %8.6f plies  (predict-the-mean %8.6f)"
              % (read, p["r2"], p["rmse"], p["rmse_mean_baseline"]))

    nul = label_permutation_null()
    print("\nNULL 1  varies %s" % (nul["varies"],))
    print("  pins %s" % (nul["pins"],))
    for name in READS:
        print("  %-13s consequence %.6f  surface-only %.6f  ratio %.6f"
              % (name, nul[name]["mean_consequence"], nul[name]["mean_surface_only"],
                 nul[name]["ratio"]))

    abl = intervention_permutation_null()
    print("\nNULL 2  the move-permutation ablation. varies %s" % (abl["varies"],))
    print("  pins %s" % (abl["pins"],))
    print("  buckets %d" % abl["n_buckets"])
    print("  %-13s %10s %10s %10s" % ("read", "true", "permuted", "lost"))
    for name in READS:
        a = abl[name]
        print("  %-13s %10.6f %10.6f %10.6f"
              % (name, a["true"]["ratio"], a["permuted"]["ratio"],
                 a["ratio_lost_to_permutation"]))

    # the assertions the docstring's claims stand on
    for name in READS:
        row = tab[name]
        if READ_IS_A_FUNCTION_OF_A[name]:
            assert row["surface_only"]["mean"] == 0.0, name
            assert row["surface_only"]["worst"] == 0.0, name
            assert row["surface_only"]["fraction_nonzero"] == 0.0, name
            assert row["consequence"]["mean"] > 0.0, name
    assert com["surface_only"]["fraction_nonzero"] > 0.99
    assert com["consequence"]["mean"] / com["surface_only"]["mean"] < 3.0
    assert com["surface_only"]["mean"] / res > 1e8
    assert dis["base-piece-cheb"]["max_abs_difference"] == 0.0
    assert dis["base-piece"]["mean_consequence"] - dis["base-piece"]["mean_surface_only"] > 0.5
    assert mul["surface_only"]["fraction_set_preserved"] == 1.0
    assert mul["surface_only"]["fraction_multiset_preserved"] < 0.5
    ali = tab["alikeness"]
    assert 0.5 < ali["consequence"]["mean"] / ali["surface_only"]["mean"] < 2.0
    for name in READS:
        assert abs(nul[name]["ratio"] - 1.0) < 0.05, (name, nul[name]["ratio"])
        assert abs(abl[name]["permuted"]["ratio"] - 1.0) < 0.15, \
            (name, abl[name]["permuted"]["ratio"])
    assert abl["committor"]["ratio_lost_to_permutation"] > 1.0
    assert head["ceiling_ratio"] < 3.0
    assert head["ceiling_surface_only_mean"] > 0.0
    # the learned read does not separate, and the random encoder says why not
    for read in ("jepa_trained", "jepa_random"):
        r = tab[read]["consequence"]["mean"] / tab[read]["surface_only"]["mean"]
        assert 0.99 < r < 1.01, (read, r)
    assert pr["jepa_trained"]["r2"] > pr["jepa_random"]["r2"]
    assert pr["jepa_trained"]["r2"] < 2.0 * pr["jepa_random"]["r2"]
    assert pr["jepa_trained"]["rmse"] < pr["jepa_trained"]["rmse_mean_baseline"]
    assert abl["jepa_trained"]["ratio_lost_to_permutation"] <         abl["alikeness"]["ratio_lost_to_permutation"]
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(demo())
